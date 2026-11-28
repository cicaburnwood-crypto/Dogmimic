#!/usr/bin/env bash
set -euo pipefail

stage=/home/m1/dogmimic-uni-exp-3a27ea5132a1
release="$stage/.deployment/contact-turn-fa09fc625a62"
startup="$stage/.deployment/classic-posture-58b33a3d319f/activate-classic-posture.sh"
unit=dg_fsm.service
library="$stage/build/robot/libhardware_library.so"
live="$stage/models/uni_exp.onnx"
candidate="$release/uni-exp-conditional.onnx"
backup="$release/previous-5fcca3909d58.onnx"
previous_hash=5fcca3909d58a0cd0223ca3e46f90d8e3fcafb73200008ef61d2d8b6fd0f3992
candidate_hash=1b1ec7244ed1e4759f1293f0066afc0884e508e824586ed5454a702586e39c75
library_hash=76f0575646762fabd16cdec13764351f939d5cb838d211085f96062a8911bdcb
incoming=''

publish() {
    printf '%s\n' "$1" > "$release/.activation.result"
    mv -T -- "$release/.activation.result" "$release/activation.result"
    printf '%s\n' "$1"
}

cleanup() {
    local status=$?
    if [[ -n "$incoming" ]]; then rm -f -- "$incoming"; fi
    if ((status)) && [[ $(cat "$release/activation.result" 2>/dev/null || true) == running ]]; then
        publish "activation stopped ($status)"
    fi
}

check_hash() { printf '%s  %s\n' "$1" "$2" | sha256sum -c -; }

observation() {
    timeout 6s "$stage/.deployment/read-stand" 2>/dev/null |
        awk '/^OBSERVATION / && /joints/ {print; found=1} END {exit !found}'
}

folded() {
    local sample
    sample=$(observation) || return 1
    printf '%s\n' "$sample"
    awk '
        {for(i=2;i<NF;i+=2) s[$i]=$(i+1)}
        END {exit !(s["joints"]==16 && s["imu"]==1 && s["power_valid"]==1 &&
                    "prior_motor_kp_max" in s && s["prior_motor_kp_max"]==0 &&
                    "leg_speed_rad_s" in s && "wheel_speed_rad_s" in s &&
                    s["standard_q_error_rad"]>.55 && s["leg_speed_rad_s"]<.25 && s["wheel_speed_rad_s"]<.25)}' <<< "$sample"
}

prepare_replacement() {
    local source=$1 digest=$2
    incoming=$(mktemp "$stage/models/.uni-exp.XXXXXXXX") || return 1
    cp -- "$source" "$incoming" && chmod --reference="$live" "$incoming" && check_hash "$digest" "$incoming"
}

swap_model() {
    mv -T -- "$incoming" "$live" || return 1
    incoming=''
}

verify_service() {
    local old_pid=$1 digest=$2 pid inode
    "$startup" --check-current || return 1
    pid=$(systemctl show "$unit" -p MainPID --value) || return 1
    [[ "$pid" =~ ^[1-9][0-9]*$ && "$pid" != "$old_pid" ]] || return 1
    check_hash "$digest" "$live" && check_hash "$library_hash" "$library" || return 1
    inode=$(stat -c %i "$library") || return 1
    sudo -n awk -v inode="$inode" -v file="$library" '$5==inode && $6==file {ok=1} END {exit !ok}' "/proc/$pid/maps" || return 1
    printf 'Original remote-control library loaded by PID %s\n' "$pid"
}

exec 9>"$stage/.deployment/uni-exp-model.lock"
flock -n 9
trap cleanup EXIT
publish running
cd "$stage/scripts"
test "$(systemctl show "$unit" -p WorkingDirectory --value)" = "$stage/scripts"
systemctl is-active --quiet "$unit"
systemctl is-active --quiet dogmimic-m1-debug.service
test -f "$live" && test ! -L "$live"
check_hash "$candidate_hash" "$candidate"
check_hash "$library_hash" "$library"
grep -Fq 'STAND_HANDOFF_OK first_torque_jump_nm 0 ' "$release/stand-replay.log"
grep -Fq 'CONTROLLER_REPLAY_OK frames 3876 ' "$release/arm-turn-replay.log"
grep -Fq 'CONTROLLER_REPLAY_OK frames 3605 ' "$release/arm-terrain-replay.log"

current_hash=$(sha256sum "$live" | cut -d ' ' -f 1)
if [[ "$current_hash" == "$candidate_hash" ]]; then
    verify_service 0 "$candidate_hash"
    publish ready
    exit 0
fi
test "$current_hash" = "$previous_hash"
if [[ ! -e "$backup" ]]; then
    incoming=$(mktemp "$release/.previous.XXXXXXXX")
    cp -p -- "$live" "$incoming"
    check_hash "$previous_hash" "$incoming"
    mv -T -- "$incoming" "$backup"
    incoming=''
fi
check_hash "$previous_hash" "$backup"
pid_before=$(systemctl show "$unit" -p MainPID --value)
prepare_replacement "$candidate" "$candidate_hash"
if ! folded; then
    publish 'waiting for folded stationary position'
    exit 2
fi
test "$(systemctl show "$unit" -p MainPID --value)" = "$pid_before"
swap_model

if sudo -n systemctl restart "$unit" && verify_service "$pid_before" "$candidate_hash"; then
    for attempt in 1 2 3; do
        if sample=$(observation); then
            printf '%s\n' "$sample"
            if awk '
                {for(i=2;i<NF;i+=2) s[$i]=$(i+1)}
                END {exit !(s["joints"]==16 && s["imu"]==1 && s["power_valid"]==1)}' <<< "$sample" &&
                systemctl is-active --quiet dogmimic-m1-debug.service; then
                publish ready
                exit 0
            fi
        fi
        sleep .5
    done
    publish 'new model loaded, hardware or logger verification pending'
    exit 3
fi

printf '%s\n' 'New runtime did not become ready. Checking posture before rollback.'
case $(systemctl show "$unit" -p ActiveState --value) in
    active|activating|reloading)
        if ! folded; then
            publish 'startup verification pending, robot state prevents rollback restart'
            exit 4
        fi
        ;;
esac
rollback_pid=$(systemctl show "$unit" -p MainPID --value)
prepare_replacement "$backup" "$previous_hash"
swap_model
if sudo -n systemctl restart "$unit" && verify_service "$rollback_pid" "$previous_hash"; then
    publish 'rolled back to previous model'
else
    publish 'previous model restored on disk, service check pending'
fi
exit 5
