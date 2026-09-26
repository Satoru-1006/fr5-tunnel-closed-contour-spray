from __future__ import annotations

import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.official_fairino import load_official_fr5_robot
from scripts.run_fr5_invert_variable_section import section

plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',type=Path,default=Path(r'C:\Users\86198\Desktop\ww\variable_invert_tcp_poses.csv'))
    ap.add_argument('--out',type=Path,default=Path(r'C:\Users\86198\Desktop\ww\variable_invert_motion.gif'))
    ap.add_argument('--frames',type=int,default=900)
    ap.add_argument('--fps',type=int,default=20)
    ap.add_argument('--start-time',type=float,default=None)
    ap.add_argument('--duration',type=float,default=None)
    args=ap.parse_args()
    with args.csv.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    q=np.asarray([[float(r[f'j{i}']) for i in range(1,7)] for r in rows])
    times=np.asarray([float(r.get('time_s',i*.04)) for i,r in enumerate(rows)])
    robot,_=load_official_fr5_robot(ROOT); robot=robot.with_base([0,-.25,.20],yaw=np.pi)
    tool=np.eye(4); tool[:3,3]=[0,0,.15]
    start=times[0] if args.start_time is None else max(times[0],args.start_time)
    end=times[-1] if args.duration is None else min(times[-1],start+args.duration)
    render_times=np.linspace(start,end,min(args.frames,len(q)))
    idx=np.unique(np.searchsorted(times,render_times,side='left').clip(0,len(q)-1))
    frames=[robot.all_joint_frames(q[i]) for i in idx]
    tcp=np.asarray([(robot.fk(q[i])@tool)[:3,3] for i in idx])
    fig=plt.figure(figsize=(8.8,6.6),facecolor='white'); ax=fig.add_subplot(111,projection='3d')
    for yy in np.linspace(0,.45,5):
        u=yy/.45; w=.78+.08*np.sin(2*np.pi*u); h=.76+.06*np.cos(2*np.pi*u); inv=.03+.015*np.sin(np.pi*u)
        p,_=section(w,h,inv,120); p2,_=section(w-.10,h-.10,inv,120)
        p=np.vstack([p,p[0]]); p2=np.vstack([p2,p2[0]])
        ax.plot(p[:,0],np.full(len(p),yy),p[:,1]+.14,color='#aaaaaa',lw=.8,alpha=.55)
        ax.plot(p2[:,0],np.full(len(p2),yy),p2[:,1]+.19,color='#d32f2f',lw=1.6,alpha=.85)
    arm,=ax.plot([],[],[],color='#202124',lw=3.2,marker='o',ms=3)
    tool_line,=ax.plot([],[],[],color='#7b1fa2',lw=4)
    tcp_dot,=ax.plot([],[],[],marker='o',color='#2e7d32',ms=6)
    info=ax.text2D(.02,.96,'',transform=ax.transAxes,va='top')
    ax.set_xlim(-.50,.50); ax.set_ylim(-.08,.50); ax.set_zlim(.05,1.0)
    ax.set_xlabel('X/m'); ax.set_ylabel('Y/m'); ax.set_zlabel('Z/m')
    ax.set_title('FR5 带仰拱变截面隧道喷涂运动（Ruckig平滑+碰撞门禁）')
    ax.view_init(24,-63)
    def update(k):
        pts=frames[k][:,:3,3]; flange=pts[-1]; t=tcp[k]
        arm.set_data(pts[:,0],pts[:,1]); arm.set_3d_properties(pts[:,2])
        tool_line.set_data([flange[0],t[0]],[flange[1],t[1]]); tool_line.set_3d_properties([flange[2],t[2]])
        tcp_dot.set_data([t[0]],[t[1]]); tcp_dot.set_3d_properties([t[2]])
        info.set_text(f'frame {k+1}/{len(frames)}')
        return arm,tool_line,tcp_dot,info
    ani=FuncAnimation(fig,update,frames=len(frames),interval=1000/args.fps,blit=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    ani.save(args.out,writer=PillowWriter(fps=args.fps)); plt.close(fig)
    print(f'Wrote {args.out} ({args.out.stat().st_size} bytes, {len(frames)} frames)')

if __name__=='__main__': main()
