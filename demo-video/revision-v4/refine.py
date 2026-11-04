from pathlib import Path
p=Path(__file__).with_name('manual_v4.py');s=p.read_text()
s=s.replace('CARTONS=[\n','CARTONS=[\n(8.75,(201,156,241,182),(145,174,164,195),(168,222,195,248),None,(199,233,219,271),(220,239,244,263)),\n')
s=s.replace('(8.8,(210,158,238,181),(169,175,181,192),(195,220,211,239),None,(201,228,218,260),(222,235,240,253)),','(8.8,(228,158,266,183),(175,174,194,195),(195,222,220,248),None,(224,232,246,271),(250,239,274,263)),')
s=s.replace('(12.2,(122,85,191,144),(3,132,38,172),(52,272,100,311),None,','(12.2,(122,85,191,144),(3,132,38,172),(3,272,53,318),None,')
# Preserve actual held source frames instead of assuming a constant camera speed.
insert='''
CARTONS.append((12.15,*next(z[1:] for z in CARTONS if z[0]==12.1)))
CARTONS.append((12.25,*next(z[1:] for z in CARTONS if z[0]==12.2)))
CARTONS.append((12.45,*next(z[1:] for z in CARTONS if z[0]==12.4)))
CARTONS.sort(key=lambda z:z[0])
TOP_PRINT=[(8.75,174,146,202,163),(9,276,145,303,161),(9.25,386,145,412,165),(9.5,450,138,486,158),(9.75,484,108,521,132),(10,511,96,554,119),(10.25,507,76,566,105),(10.5,500,64,565,84),(10.75,481,64,556,93),(11,482,64,556,89),(11.25,482,64,555,90),(11.5,461,64,534,89),(11.75,313,64,379,90),(12,208,64,262,96),(12.1,129,64,181,94),(12.15,129,64,181,94),(12.2,78,64,126,89),(12.25,78,64,126,89),(12.3,71,65,116,105),(12.4,35,64,74,85),(12.5,-60,64,-5,85)]
TOP_RETURN=[(13.6,99,64,172,86),(13.65,170,64,244,86),(13.7,326,64,399,89),(13.75,394,64,468,87),(13.8,553,64,629,87),(13.85,563,64,643,87),(13.9,643,64,719,87)]
'''
s=s.replace('# The entire visible large quadruped',insert+'\n# The entire visible large quadruped')
s=s.replace("for k in [DOG1,DOG2]:active_rect(k,t,coarse,(3,64,637,426),pad=4)","for k in [TOP_PRINT,TOP_RETURN]:active_rect(k,t,fine,(3,64,637,426),pad=5)\n   for k in [DOG1,DOG2]:active_rect(k,t,coarse,(3,64,637,426),pad=12)")
s=s.replace('active_rect([[t,*rect]],t,fine,(3,64,637,426),pad=5)','active_rect([[t,*rect]],t,fine,(3,64,637,426),pad=12 if not 11.6<t<12.5 else 18)')
s=s.replace('active_rect(POV,t,coarse,(1094,72,1636,480),pad=5)','active_rect(POV,t,coarse,(1094,72,1636,480),pad=12)')
s=s.replace('active_rect(GLASS,t,coarse,(284,0,1094,1080),pad=5)','active_rect(GLASS,t,coarse,(284,0,1094,1080),pad=17)')
fix='''
def replace_anchors(keys,updates):
 d={z[0]:z for z in keys};d.update({z[0]:z for z in updates});keys[:]=[d[t] for t in sorted(d)]
replace_anchors(POV,[(5.375,1360,127,1558,309),(5.625,1251,143,1461,314),(5.875,1230,140,1438,321),(6.125,1177,119,1390,315),(6.375,1020,149,1279,343),(6.6,920,142,1182,359),(6.65,903,142,1182,363),(6.7,886,142,1138,365),(6.75,871,142,1122,366),(6.8,856,142,1114,369),(6.85,842,142,1095,370),(6.9,830,142,1076,371)])
replace_anchors(GLASS,[(7.25,850,324,1081,618),(7.75,737,319,958,619),(8.25,677,321,912,629),(8.75,683,319,919,634),(9.25,567,307,809,642),(9.75,420,310,668,654),(10.25,313,316,557,659),(10.75,405,312,638,648),(11.25,379,275,621,625),(11.75,389,287,622,637),(12.25,399,306,624,655),(12.75,367,228,585,592)])
'''
s=s.replace('\ndef active_rect(',fix+'\ndef active_rect(')
s=s.replace('(7.25,997,533,32,29,962,550,1042,660)','(7.25,977,532,32,29,942,550,1022,660)').replace('(8.25,919,542,32,31,877,562,962,660)','(8.25,888,538,32,31,847,558,932,660)').replace('(8.75,922,537,32,30,886,557,973,660)','(8.75,942,539,32,30,905,559,993,660)')
old='''   cv2.ellipse(m,(round(cx),round(cy)),(round(rx),round(ry)),0,0,360,0,-1)
   pts=np.array([[cx-19,cy+18],[cx+20,cy+20],[x1,y0+24],[x1,y1],[x0,y1],[x0,y0+30]],np.int32)
   cv2.fillPoly(m,[pts],0)'''
new='''   xa,ya,xb,yb=max(284,round(cx-70)),max(0,round(cy-45)),min(1094,round(cx+80)),round(cy+130)
   q=im[ya:yb,xa:xb];dark=q.max(2)<115
   white=((q.min(2)>175)&((q.max(2).astype(int)-q.min(2))<55)).astype(np.uint8)
   thick=cv2.morphologyEx(white,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))>0
   m[ya:yb,xa:xb][dark|thick]=0'''
assert old in s;s=s.replace(old,new)
# Foreground geometries may only preserve thick opaque material, not fine glass strokes.
s=s.replace("   cv2.line(m,(round(ax)","   protect=np.zeros_like(m)\n   cv2.line(protect,(round(ax)")
s=s.replace("(round(bx),round(by)),0,29)","(round(bx),round(by)),255,29)")
s=s.replace("cv2.line(m,(round(px)","cv2.line(protect,(round(px)").replace("(round(px),round(pb)),0,12)","(round(px),round(pb)),255,12)")
s=s.replace("cv2.rectangle(m,(round(px-17)","cv2.rectangle(protect,(round(px-17)").replace("(round(px+17),round(py+18)),0,-1)","(round(px+17),round(py+18)),255,-1)")
s=s.replace("cv2.rectangle(m,(round(px-16)","cv2.rectangle(protect,(round(px-16)").replace("(round(px+16),round(pb+30)),0,-1)","(round(px+16),round(pb+30)),255,-1)\n   bright=((im.min(2)>175)&((im.max(2).astype(int)-im.min(2))<55)).astype(np.uint8)\n   thick=cv2.morphologyEx(bright,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))>0\n   dark=im.max(2)<125\n   red=(im[:,:,2].astype(int)-im[:,:,1].astype(int)>45)\n   m[(protect>0)&(thick|dark|red)]=0")
s=s.replace('(hsv[:,:,1]>60)','(hsv[:,:,1]>85)')
s=s.replace('   dark=(q.max(2)<105)','   skin=cv2.morphologyEx(skin.astype(np.uint8),cv2.MORPH_OPEN,np.ones((7,7),np.uint8))>0\n   dark=(q.max(2)<105)')
# Keep the original speed glyphs crisp where a redaction crosses their badge.
s=s.replace('  return out\n','''  if self.c==3 and (fine[74:192,14:178].any() or coarse[74:192,14:178].any()):
   t=f/60;key=('6X' if t>=12.7 else '3X') if 8.7<t<14.1 else ('2X' if 48<t<52 else '4X')
   region=out[74:192,14:178];region[:]=24
   region[self.speed[key]]=im[74:192,14:178][self.speed[key]]
  return out
''')
p.write_text(s)
