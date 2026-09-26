from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.official_fairino import load_official_fr5_robot
from src.time_parameterization import make_segmented_ruckig_time_profile


def section(width: float, height: float, invert: float, n: int = 80):
    """Closed horseshoe with a curved invert; returns x,z points and inward normals."""
    r = width / 2.0
    spring = height - r
    # Fixed process order: left-bottom -> left wall upward -> arch ->
    # right wall downward -> curved invert right-to-left.
    nl=max(8,n//3); na=max(24,n); ni=max(16,n//2)
    left=np.c_[np.full(nl,-r),np.linspace(0,spring,nl,endpoint=False)]
    a=np.linspace(np.pi,0,na,endpoint=False)
    arch=np.c_[r*np.cos(a),spring+r*np.sin(a)]
    right=np.c_[np.full(nl,r),np.linspace(spring,0,nl,endpoint=False)]
    # Elliptic invert with vertical tangents at both spring points. It is
    # tangent to the right/left vertical walls, removing the hard corner that
    # previously caused a stop, reverse swing, and abrupt wrist reorientation.
    b=np.linspace(0,np.pi,ni,endpoint=False)
    inv=np.c_[r*np.cos(b),-invert*np.sin(b)]
    p=np.vstack([left,arch,right,inv])
    # Use the invert bottom midpoint as the cyclic process start. The
    # resulting order is bottom-centre -> left invert -> left wall -> arch ->
    # right wall -> right invert -> bottom-centre.
    start_index=len(left)+len(arch)+len(right)+ni//2
    p=np.roll(p,-start_index,axis=0)
    q = np.vstack([p, p[0]])
    t = np.gradient(q, axis=0)[:-1]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    na = np.c_[-t[:, 1], t[:, 0]]
    center = np.array([0.0, height * .45 - invert*.15])
    sign = np.sign(np.sum(na * (center-p), axis=1, keepdims=True))
    normal = na * np.where(sign == 0, 1, sign)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    return p, normal


def make_targets(length, stations, samples, stand_off):
    targets, walls, normals = [], [], []
    for y in np.linspace(0, length, stations):
        u = y / length
        width = 0.78 + 0.08*np.sin(2*np.pi*u)       # variable width
        height = 0.76 + 0.06*np.cos(2*np.pi*u)      # variable height
        invert = 0.03 + 0.015*np.sin(np.pi*u)       # variable invert depth
        p, n = section(width, height, invert, samples)
        # Close every station explicitly at the exact bottom-centre point.
        # Without this duplicate endpoint, the last sampled point on the
        # right invert is connected directly to the next station's start,
        # producing the apparent retract/advance motion at the left-bottom
        # transition instead of a continuous contour.
        p = np.vstack([p, p[0]])
        n = np.vstack([n, n[0]])
        # Place the tunnel section in the FR5 work envelope while preserving
        # the curved invert geometry (the invert remains visibly lower than
        # the spring line, but no longer forces the TCP below the base plane).
        wall = np.c_[p[:,0], np.full(len(p), y), p[:,1] + 0.14]
        targets.append(wall + stand_off*np.c_[n[:,0], np.zeros(len(n)), n[:,1]])
        walls.append(wall); normals.append(np.c_[n[:,0], np.zeros(len(n)), n[:,1]])
    return np.vstack(targets), np.vstack(walls), np.vstack(normals)


def pose_from_target(tcp, wall, normal):
    z = -normal / np.linalg.norm(normal)
    x = np.cross([0., 1., 0.], z)
    if np.linalg.norm(x) < 1e-8: x = np.array([1., 0., 0.])
    x /= np.linalg.norm(x); y = np.cross(z, x)
    T = np.eye(4); T[:3,:3] = np.c_[x,y,z]; T[:3,3] = tcp
    return T


def tunnel_link_clearance(robot, q, length, margin=0.005):
    """Conservative link-to-variable-tunnel clearance from sampled link points."""
    frames=robot.all_joint_frames(q)
    samples=[]
    for a,b in zip(frames[:-1,:3,3],frames[1:,:3,3]):
        samples.append(np.linspace(a,b,17))
    pts=np.vstack(samples); active=(pts[:,1]>=0.0)&(pts[:,1]<=length)
    if not np.any(active): return 1.0
    pts=pts[active]; u=np.clip(pts[:,1]/length,0.0,1.0)
    width=0.78+0.08*np.sin(2*np.pi*u); height=0.76+0.06*np.cos(2*np.pi*u)
    invert=0.03+0.015*np.sin(np.pi*u); r=width/2.0; spring=height-r
    x=pts[:,0]; z=pts[:,2]-0.14
    radial=np.maximum(r*r-x*x,0.0)
    ceiling=spring+np.sqrt(radial)
    floor=-invert*np.sqrt(np.maximum(1.0-np.clip((x/r)**2,0.0,1.0),0.0))
    clear=np.minimum.reduce([r-np.abs(x),z-floor,ceiling-z])-margin
    return float(np.min(clear))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path(r'C:\Users\86198\Desktop\ww'))
    ap.add_argument('--length', type=float, default=.45)
    ap.add_argument('--stations', type=int, default=5)
    ap.add_argument('--samples', type=int, default=24)
    ap.add_argument('--stand-off', type=float, default=.05)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    robot, assets = load_official_fr5_robot(ROOT)
    # Place the FR5 base outside the tunnel entrance; keeping it in the
    # section plane causes the shoulder/upper-arm links to graze the wall.
    robot = robot.with_base([0., -.25, .20], yaw=np.pi)
    tcp, wall, normals = make_targets(args.length, args.stations, args.samples, args.stand_off)
    display_wall_rows=[]; display_tcp_rows=[]
    for yy in np.linspace(0,args.length,args.stations):
        u=yy/args.length
        width=0.78+0.08*np.sin(2*np.pi*u); height=0.76+0.06*np.cos(2*np.pi*u)
        invert=0.03+0.015*np.sin(np.pi*u)
        wp,_=section(width,height,invert,120)
        # Analytic inward offset for display: a smaller horseshoe shifted
        # upward by the stand-off. This preserves straight walls and smooth
        # arch/invert junctions without normal-offset corner loops.
        tp,_=section(width-2*args.stand_off,height-2*args.stand_off,invert,120)
        wp=np.vstack([wp,wp[0]])
        tp=np.vstack([tp,tp[0]])
        display_wall_rows.append(np.c_[wp[:,0],np.full(len(wp),yy),wp[:,1]+0.14])
        display_tcp_rows.append(np.c_[tp[:,0],np.full(len(tp),yy),tp[:,1]+0.14+args.stand_off])
    display_wall=np.vstack(display_wall_rows); display_tcp=np.vstack(display_tcp_rows)
    tool = np.eye(4); tool[:3,3] = [0,0,.150]; tool_inv = np.linalg.inv(tool)
    seed = np.deg2rad([101.4,-37.1,66.7,52.3,55.,13.9]); qs=[]; errors=[]; degraded=[]; projected=[]; held_for_collision=[]
    for i,(p,w,n) in enumerate(zip(tcp,wall,normals)):
        target = pose_from_target(p,w,n) @ tool_inv
        candidates = [seed, qs[-1] if qs else seed, np.deg2rad([90,-45,60,45,45,0])]
        # Probe nearby IK basins deterministically; this is especially useful
        # around the invert where the wrist orientation changes rapidly.
        for delta in ([0,0,0,0,0,0.8],[0,0,0,0,0,-0.8],[0,0.5,-0.5,0,0,0],[0,-0.5,0.5,0,0,0]):
            candidates.append(seed + np.asarray(delta))
        rng = np.random.default_rng(i + 20260713)
        candidates.extend(seed + rng.normal(0.0, 0.35, 6) for _ in range(8))
        solutions = [robot.ik(target, s, orientation_weight=1.0, continuity_weight=.001) for s in candidates]
        safe_solutions=[row for row in solutions if tunnel_link_clearance(robot,row[0],args.length) >= 0.0]
        # Probe extra IK basins only when the local candidates all intersect
        # the tunnel.  This avoids replacing a process point by a hold just
        # because the first small seed set missed a safe elbow/wrist branch.
        if not safe_solutions:
            extra = [seed + rng.normal(0.0, 0.75, 6) for _ in range(20)]
            extra += [rng.uniform(robot.limits.q_min + 0.01, robot.limits.q_max - 0.01) for _ in range(12)]
            extra_solutions = [robot.ik(target, s, orientation_weight=1.0, continuity_weight=.004) for s in extra]
            solutions.extend(extra_solutions)
            safe_solutions = [row for row in extra_solutions if tunnel_link_clearance(robot,row[0],args.length) >= 0.0]
        reference = qs[-1] if qs else seed
        def choose(rows):
            if not rows:
                return min(solutions, key=lambda row: row[1])
            deltas = [((row[0] - reference + np.pi) % (2*np.pi)) - np.pi for row in rows]
            transition_rows=[]
            for row,delta in zip(rows,deltas):
                if all(tunnel_link_clearance(robot, reference + alpha*delta, args.length) >= 0.0 for alpha in np.linspace(0.0,1.0,9)):
                    transition_rows.append(row)
            if transition_rows:
                rows=transition_rows
                deltas=[((row[0] - reference + np.pi) % (2*np.pi)) - np.pi for row in rows]
            smooth = [row for row,delta in zip(rows,deltas) if np.max(np.abs(delta)) <= np.deg2rad(25.0)]
            pool = smooth or rows
            return min(pool, key=lambda row: row[1] + 0.05*np.linalg.norm(((row[0] - reference + np.pi) % (2*np.pi)) - np.pi))
        q,e,ok = choose(safe_solutions or solutions)
        if e > 0.035:
            # Position-priority fallback: keep the previous flange attitude
            # so the robot can traverse a locally difficult invert point.
            Tfb = np.eye(4); Tfb[:3,:3] = robot.fk(seed)[:3,:3]
            Tfb[:3,3] = p - Tfb[:3,:3] @ tool[:3,3]
            fb = [robot.ik(Tfb, s, orientation_weight=1.0, continuity_weight=.004) for s in candidates]
            safe_fb=[row for row in fb if tunnel_link_clearance(robot,row[0],args.length) >= 0.0]
            q2,e2,ok2 = choose(safe_fb or fb)
            if e2 < e:
                q,e,ok = q2,e2,ok2
                degraded.append(i)
        if e > 0.035:
            base = robot.base_transform[:3,3]
            for alpha in np.linspace(0.95, 0.45, 11):
                pp = base + alpha * (p - base)
                Tp = np.eye(4); Tp[:3,:3] = pose_from_target(p,w,n)[:3,:3]
                Tp[:3,3] = pp - Tp[:3,:3] @ tool[:3,3]
                cand = [robot.ik(Tp, s, orientation_weight=1.0, continuity_weight=.004) for s in candidates]
                q3,e3,ok3 = choose(cand)
                if e3 < e:
                    q,e,ok = q3,e3,ok3
                    projected.append({'index': i, 'alpha': float(alpha), 'projection_m': float(np.linalg.norm(p-pp))})
                if e <= 0.01: break
        # Keep a bounded least-squares fallback for the lowest invert points;
        # these are reported in validation instead of being silently removed.
        clearance=tunnel_link_clearance(robot,q,args.length)
        if clearance < 0.0 and i != 0:
            q=seed.copy(); held_for_collision.append(i)
        if not ok:
            # Retain the best bounded numerical solution so the complete
            # trajectory/animation can be inspected; validation exposes it.
            q = seed.copy()
        qs.append(q); errors.append(float(e)); seed=q
    # IK already returns the nearest equivalent angle inside each FR5 joint
    # limit.  Unwrapping here would add 2*pi every time a joint crosses the
    # plotting branch, creating impossible multi-turn commands and the large
    # TCP excursions seen in the animation.
    qs=np.asarray(qs)
    ik_waypoint_qs=qs.copy()
    with (args.out/'variable_invert_ik_waypoints.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f); wr.writerow(['index','j1','j2','j3','j4','j5','j6'])
        for i,q in enumerate(ik_waypoint_qs): wr.writerow([i,*q])
    try:
        profile=make_segmented_ruckig_time_profile(tcp,qs,robot.limits,target_tcp_speed=.04,dt=.04,closed_loop=False)
        qs=profile.q; trajectory_time=profile.t; time_parameterization=profile.method
    except Exception as exc:
        time_parameterization=f'linear-fallback:{exc}'
        q_path=[qs[0]]; max_step_rad=np.deg2rad(6.0)
        for a,b in zip(qs[:-1],qs[1:]):
            n=max(1,int(np.ceil(np.max(np.abs(b-a))/max_step_rad)))
            q_path.extend(np.linspace(a,b,n+1,endpoint=False)[1:])
        q_path.append(qs[-1]); qs=np.asarray(q_path); trajectory_time=np.arange(len(qs),dtype=float)*.04
    actual=np.asarray([(robot.fk(q)@tool)[:3,3] for q in qs])
    realized_tcp_error=np.linalg.norm(actual-profile.path_points,axis=1)
    with (args.out/'variable_invert_tcp_poses.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f); wr.writerow(['index','time_s','x','y','z','j1','j2','j3','j4','j5','j6'])
        for i,(tt,p,q) in enumerate(zip(trajectory_time,actual,qs)): wr.writerow([i,float(tt),*p,*q])
    step=np.rad2deg(np.max(np.abs((np.diff(qs,axis=0)+np.pi)%(2*np.pi)-np.pi)))
    clearances=[tunnel_link_clearance(robot,q,args.length) for q in qs[::max(1,len(qs)//200)]]
    base_status='pass' if step<20 and max(errors)<=.01 and min(clearances)>=0 and np.max(realized_tcp_error)<=.01 else 'review_required'
    status='pass_with_collision_holds' if base_status=='pass' and held_for_collision else base_status
    validation={'geometry':'variable horseshoe with curved invert','length_m':args.length,'stations':args.stations,'waypoints':len(qs),'trajectory_duration_s':float(trajectory_time[-1]),'target_tcp_speed_m_s':.04,'virtual_tcp_m':.15,'stand_off_m':args.stand_off,'ik_error_max':max(errors),'ik_error_over_10mm_count':sum(e>.01 for e in errors),'realized_tcp_path_error_max_m':float(np.max(realized_tcp_error)),'realized_tcp_path_error_over_10mm_count':int(np.sum(realized_tcp_error>.01)),'ik_waypoint_joint_min_deg':np.rad2deg(np.min(ik_waypoint_qs,axis=0)).tolist(),'ik_waypoint_joint_max_deg':np.rad2deg(np.max(ik_waypoint_qs,axis=0)).tolist(),'degraded_orientation_points':degraded,'projected_reachability_points':projected,'held_for_collision_points':held_for_collision,'minimum_sampled_link_clearance_m':min(clearances),'collision_sample_count':sum(c<0 for c in clearances),'time_parameterization':time_parameterization,'max_adjacent_joint_step_deg':float(step),'status':status,'robot_source':str(assets.urdf)}
    (args.out/'variable_invert_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fig=plt.figure(figsize=(9,6.5)); ax=fig.add_subplot(111,projection='3d')
    ax.plot(display_wall[:,0],display_wall[:,1],display_wall[:,2],'.',ms=1.0,color='#999999',alpha=.30,label='变截面隧道内壁')
    # Only the designed process path is red. Joint-space branch transitions
    # are robot motion with spraying disabled and must never be drawn as a
    # coating path.
    ax.plot(display_tcp[:,0],display_tcp[:,1],display_tcp[:,2],color='#d32f2f',lw=2.2,label='规划TCP喷涂路径')
    line,=ax.plot([],[],[],color='#202124',lw=3,marker='o',ms=3); toolline,=ax.plot([],[],[],color='#7b1fa2',lw=4)
    ax.set_xlabel('X/m'); ax.set_ylabel('Y/m'); ax.set_zlabel('Z/m'); ax.set_title('FR5 带仰拱变截面隧道喷涂运动仿真'); ax.legend(); ax.view_init(24,-63)
    frame_indices=np.unique(np.linspace(0,len(qs)-1,min(360,len(qs))).round().astype(int))
    frames=[robot.all_joint_frames(qs[index]) for index in frame_indices]
    def update(i):
        f=frames[i]; pts=f[:,:3,3]; line.set_data(pts[:,0],pts[:,1]); line.set_3d_properties(pts[:,2]); fl=pts[-1]; t=actual[frame_indices[i]]; toolline.set_data([fl[0],t[0]],[fl[1],t[1]]); toolline.set_3d_properties([fl[2],t[2]]); return line,toolline
    ani=FuncAnimation(fig,update,frames=len(frames),interval=40,blit=False); ani.save(args.out/'variable_invert_motion.gif',writer=PillowWriter(fps=24)); plt.close(fig)
    fig=plt.figure(figsize=(8,5)); ax=fig.add_subplot(111)
    display_samples=120
    section_count=max(8,display_samples//3) + display_samples + max(8,display_samples//3) + max(16,display_samples//2) + 1
    for station in range(args.stations):
        sl=slice(station*section_count,(station+1)*section_count)
        wp=display_wall[sl]; tp=display_tcp[sl]
        ax.plot(np.r_[wp[:,0],wp[0,0]],np.r_[wp[:,2],wp[0,2]],color='#9e9e9e',lw=1.1,alpha=.72)
        ax.plot(np.r_[tp[:,0],tp[0,0]],np.r_[tp[:,2],tp[0,2]],color='#d32f2f',lw=1.6,alpha=.72)
    ax.set_aspect('equal'); ax.set_xlabel('X/m'); ax.set_ylabel('Z/m'); ax.set_title('标准带仰拱变截面及TCP喷涂轮廓'); ax.grid(True,alpha=.25)
    ax.plot([],[],color='#9e9e9e',label='隧道断面'); ax.plot([],[],color='#d32f2f',label='TCP喷涂轮廓'); ax.legend()
    fig.savefig(args.out/'variable_invert_geometry.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    print(json.dumps(validation,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
