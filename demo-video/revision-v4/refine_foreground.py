from pathlib import Path
p=Path(__file__).with_name('manual_v4.py');s=p.read_text()
a=s.index('   xa,ya,xb,yb=max(284,round(cx-70))');b=s.index('  if 9.25<=t<=13.4:',a)
s=s[:a]+'''   xa,ya,xb,yb=max(284,round(cx-90)),max(0,round(cy-60)),min(1094,round(cx+105)),round(cy+165)
   q=im[ya:yb,xa:xb].copy();gm=np.zeros(q.shape[:2],np.uint8)
   cv2.ellipse(gm,(round(cx-xa),round(cy-ya)),(44,42),0,0,360,cv2.GC_PR_FGD,-1)
   pts=np.array([[cx-26,cy+16],[cx+24,cy+12],[cx+78,cy+67],[cx+82,cy+164],[cx-64,cy+164],[cx-50,cy+57]],np.int32)-[xa,ya]
   cv2.fillPoly(gm,[pts],cv2.GC_PR_FGD)
   cv2.ellipse(gm,(round(cx-xa),round(cy-ya)),(16,14),0,0,360,cv2.GC_FGD,-1)
   sy,sx=round(cy+75-ya),round(cx+33-xa)
   if 4<sy<q.shape[0]-4 and 4<sx<q.shape[1]-4 and np.median(q[sy-3:sy+4,sx-3:sx+4])>170:
    cv2.circle(gm,(sx,sy),4,cv2.GC_FGD,-1)
   cv2.grabCut(q,gm,None,np.zeros((1,65),float),np.zeros((1,65),float),3,cv2.GC_INIT_WITH_MASK)
   fg=((gm==cv2.GC_FGD)|(gm==cv2.GC_PR_FGD)).astype(np.uint8)
   fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
   m[ya:yb,xa:xb][fg>0]=0
''' +s[b:]
a=s.index('   protect=np.zeros_like(m)');b=s.index('  if 9.25<=t<=14.25:',a)
s=s[:a]+'''   cv2.line(m,(round(ax),round(ay)),(round(bx),round(by)),0,16)
   cv2.line(m,(round(px),round(py+16)),(round(px),round(pb)),0,5)
   cv2.rectangle(m,(round(px-10),round(py+3)),(round(px+10),round(py+16)),0,-1)
   cv2.rectangle(m,(round(px-10),round(pb-45)),(round(px+10),round(pb+20)),0,-1)
''' +s[b:]
s=s.replace('   ax,ay,bx,by,px,py,pb=interp(keys,t)','''   replace_anchors(keys,[(9.75,351,469,588,234,501,381,620),(10.25,256,463,480,245,388,379,634),(10.75,334,466,563,238,478,378,638),(11.25,365,433,541,231,503,363,600),(11.75,371,443,545,241,517,370,610),(12.25,373,478,563,260,515,385,640),(12.75,406,394,581,211,530,314,564)])
   ax,ay,bx,by,px,py,pb=interp(keys,t)''')
s=s.replace('   dark=(q.max(2)<105)','   dark=(q.max(2)<105) if t>=13.2 else np.zeros(q.shape[:2],bool)')
p.write_text(s)
