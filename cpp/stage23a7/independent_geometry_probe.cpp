#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct V {
  double x = 0, y = 0, z = 0;
  V operator+(const V& o) const { return {x + o.x, y + o.y, z + o.z}; }
  V operator-(const V& o) const { return {x - o.x, y - o.y, z - o.z}; }
  V operator*(double s) const { return {x * s, y * s, z * s}; }
};
static V cross(const V& a, const V& b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
static double dot(const V& a, const V& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static double norm2(const V& a) { return dot(a,a); }
static double norm(const V& a) { return std::sqrt(norm2(a)); }
static V normalize(const V& a) { double n=norm(a); return n > 0 ? a*(1.0/n) : V{}; }

struct M {
  double a[4][4]{};
  static M identity() { M t; for (int i=0;i<4;++i) t.a[i][i]=1; return t; }
};
static M operator*(const M& x, const M& y) {
  M z; for (int i=0;i<4;++i) for (int j=0;j<4;++j) for (int k=0;k<4;++k) z.a[i][j]+=x.a[i][k]*y.a[k][j]; return z;
}
static V apply(const M& t, const V& v) { return {t.a[0][0]*v.x+t.a[0][1]*v.y+t.a[0][2]*v.z+t.a[0][3], t.a[1][0]*v.x+t.a[1][1]*v.y+t.a[1][2]*v.z+t.a[1][3], t.a[2][0]*v.x+t.a[2][1]*v.y+t.a[2][2]*v.z+t.a[2][3]}; }
static M transl(const V& p) { M t=M::identity(); t.a[0][3]=p.x; t.a[1][3]=p.y; t.a[2][3]=p.z; return t; }
static M rx(double q) { M t=M::identity(); double c=std::cos(q),s=std::sin(q); t.a[1][1]=c;t.a[1][2]=-s;t.a[2][1]=s;t.a[2][2]=c; return t; }
static M rz(double q) { M t=M::identity(); double c=std::cos(q),s=std::sin(q); t.a[0][0]=c;t.a[0][1]=-s;t.a[1][0]=s;t.a[1][1]=c; return t; }

struct Tri { int i[3]{}; };
struct Mesh { std::vector<V> v; std::vector<Tri> t; };
struct Box {
  V lo{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
  V hi{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()};
  void add(const V& p) { lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z); }
  bool valid() const { return lo.x<=hi.x && lo.y<=hi.y && lo.z<=hi.z; }
};
static double box_dist2(const Box& a, const Box& b) {
  double d[3]{}; const double al[3]={a.lo.x,a.lo.y,a.lo.z}, ah[3]={a.hi.x,a.hi.y,a.hi.z}, bl[3]={b.lo.x,b.lo.y,b.lo.z}, bh[3]={b.hi.x,b.hi.y,b.hi.z};
  for (int k=0;k<3;++k) d[k]=ah[k]<bl[k]?bl[k]-ah[k]:bh[k]<al[k]?al[k]-bh[k]:0; return d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
}
struct BVHNode { Box local, world; int left=-1,right=-1; std::vector<int> tris; };
struct BVH {
  const Mesh* mesh=nullptr; std::vector<BVHNode> n;
  int build(std::vector<int> ids) {
    BVHNode q; for (int id:ids) for (int x:mesh->t[id].i) q.local.add(mesh->v[x]);
    int me=n.size(); n.push_back(q); if (ids.size()<=8) { n[me].tris=std::move(ids); return me; }
    V ext=q.local.hi-q.local.lo; int ax=(ext.y>ext.x && ext.y>=ext.z)?1:(ext.z>ext.x && ext.z>=ext.y?2:0);
    std::sort(ids.begin(),ids.end(),[&](int a,int b){ V ca=(mesh->v[mesh->t[a].i[0]]+mesh->v[mesh->t[a].i[1]]+mesh->v[mesh->t[a].i[2]])*(1.0/3); V cb=(mesh->v[mesh->t[b].i[0]]+mesh->v[mesh->t[b].i[1]]+mesh->v[mesh->t[b].i[2]])*(1.0/3); return (ax==0?ca.x:ax==1?ca.y:ca.z)<(ax==0?cb.x:ax==1?cb.y:cb.z); });
    size_t mid=ids.size()/2; std::vector<int> l(ids.begin(),ids.begin()+mid),r(ids.begin()+mid,ids.end()); n[me].left=build(std::move(l)); n[me].right=build(std::move(r)); return me;
  }
  void init(const Mesh& m) { mesh=&m; n.clear(); std::vector<int> ids(m.t.size()); for(size_t i=0;i<ids.size();++i)ids[i]=int(i); build(std::move(ids)); }
  Box update(int ix,const std::vector<V>& w) { auto& q=n[ix]; Box b; if(q.left<0){ for(int id:q.tris) for(int x:mesh->t[id].i) b.add(w[x]); } else { b=update(q.left,w); Box r=update(q.right,w); b.add(r.lo);b.add(r.hi); } q.world=b; return b; }
};

static Mesh read_stl(const fs::path& p) {
  std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error("mesh open failed: "+p.string()); std::string data((std::istreambuf_iterator<char>(f)),{}); Mesh m;
  bool binary=data.size()>=84; uint32_t n=0; if(binary) { std::memcpy(&n,data.data()+80,4); binary=(data.size()==84ull+50ull*n); }
  if(binary) { m.v.reserve(3ull*n); for(uint32_t i=0;i<n;++i){ const unsigned char* q=(const unsigned char*)data.data()+84ull+50ull*i+12; float x[9]; std::memcpy(x,q,36); int base=m.v.size(); for(int k=0;k<9;k+=3)m.v.push_back({x[k],x[k+1],x[k+2]}); m.t.push_back({{base,base+1,base+2}}); } }
  else { std::istringstream in(data); std::string w; std::vector<V> tri; while(in>>w){ if(w=="vertex"){V v;in>>v.x>>v.y>>v.z;tri.push_back(v);if(tri.size()==3){int b=m.v.size();m.v.insert(m.v.end(),tri.begin(),tri.end());m.t.push_back({{b,b+1,b+2}});tri.clear();}} } }
  if(m.t.empty()) throw std::runtime_error("mesh has no triangles: "+p.string()); return m;
}

struct Closest { double d2=std::numeric_limits<double>::infinity(); V a{},b{}; int ia=-1,ib=-1; bool intersect=false; };
static bool point_tri(const V&p,const V&a,const V&b,const V&c,V& q) { V ab=b-a,ac=c-a,n=cross(ab,ac); double nn=norm2(n); if(nn<1e-28)return false; double plane=dot(n,p-a); if(std::abs(plane)>1e-8*std::sqrt(nn))return false; V ap=p-a; double d00=dot(ab,ab),d01=dot(ab,ac),d11=dot(ac,ac),d20=dot(ap,ab),d21=dot(ap,ac),den=d00*d11-d01*d01; if(std::abs(den)<1e-28)return false; double u=(d11*d20-d01*d21)/den,v=(d00*d21-d01*d20)/den; if(u>=-1e-9&&v>=-1e-9&&u+v<=1+1e-9){q=a+ab*u+ac*v;return true;}return false; }
static bool seg_tri(const V&p,const V&q,const V&a,const V&b,const V&c,V& hit) { V d=q-p,e1=b-a,e2=c-a,h=cross(d,e2);double den=dot(e1,h); if(std::abs(den)<1e-12)return false;double inv=1/den,s=dot(p-a,h)*inv; if(s<-1e-9||s>1+1e-9)return false; V u=cross(p-a,e1);double v=dot(d,u)*inv; if(v<-1e-9||s+v>1+1e-9)return false;double z=dot(e2,u)*inv;if(z<-1e-9||z>1+1e-9)return false;hit=p+d*z;return true; }
static double seg_seg(const V&p1,const V&q1,const V&p2,const V&q2,V& a,V&b) { V d1=q1-p1,d2=q2-p2,r=p1-p2;double A=dot(d1,d1),E=dot(d2,d2),F=dot(d2,r),s=0,t=0; if(A<=1e-28&&E<=1e-28){a=p1;b=p2;return norm2(a-b);} if(A<=1e-28){t=std::clamp(F/E,0.0,1.0);} else {double C=dot(d1,r);if(E<=1e-28)s=std::clamp(-C/A,0.0,1.0);else{double B=dot(d1,d2),den=A*E-B*B;if(den!=0)s=std::clamp((B*F-C*E)/den,0.0,1.0);double tnom=B*s+F;if(tnom<0){t=0;s=std::clamp(-C/A,0.0,1.0);}else if(tnom>E){t=1;s=std::clamp((B-C)/A,0.0,1.0);}else t=tnom/E;}}a=p1+d1*s;b=p2+d2*t;return norm2(a-b); }
static double point_tri_dist(const V&p,const V&a,const V&b,const V&c,V& q) { V ab=b-a,ac=c-a,ap=p-a;double d1=dot(ab,ap),d2=dot(ac,ap);if(d1<=0&&d2<=0){q=a;return norm2(p-q);}V bp=p-b;double d3=dot(ab,bp),d4=dot(ac,bp);if(d3>=0&&d4<=d3){q=b;return norm2(p-q);}double vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0){double v=d1/(d1-d3);q=a+ab*v;return norm2(p-q);}V cp=p-c;double d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0&&d5<=d6){q=c;return norm2(p-q);}double vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0){double w=d2/(d2-d6);q=a+ac*w;return norm2(p-q);}double va=d3*d6-d5*d4;if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){V bc=c-b;double w=(d4-d3)/((d4-d3)+(d5-d6));q=b+bc*w;return norm2(p-q);}double den=1.0/(va+vb+vc),v=vb*den,w=vc*den;q=a+ab*v+ac*w;return norm2(p-q); }
static Closest tri_pair(const V A[3],const V B[3],int ia,int ib) { Closest r; V h; for(int e=0;e<3;++e){if(seg_tri(A[e],A[(e+1)%3],B[0],B[1],B[2],h)){r.d2=0;r.a=h;r.b=h;r.ia=ia;r.ib=ib;r.intersect=true;return r;}if(seg_tri(B[e],B[(e+1)%3],A[0],A[1],A[2],h)){r.d2=0;r.a=h;r.b=h;r.ia=ia;r.ib=ib;r.intersect=true;return r;}} if(point_tri(A[0],B[0],B[1],B[2],h)){r.d2=0;r.a=A[0];r.b=h;r.ia=ia;r.ib=ib;r.intersect=true;return r;} if(point_tri(B[0],A[0],A[1],A[2],h)){r.d2=0;r.a=h;r.b=B[0];r.ia=ia;r.ib=ib;r.intersect=true;return r;}
  auto take=[&](double d,const V&a,const V&b){if(d<r.d2){r.d2=d;r.a=a;r.b=b;r.ia=ia;r.ib=ib;}};for(int e=0;e<3;++e){V a,b;take(seg_seg(A[e],A[(e+1)%3],B[0],B[1],a,b),a,b);take(seg_seg(A[e],A[(e+1)%3],B[1],B[2],a,b),a,b);take(seg_seg(A[e],A[(e+1)%3],B[2],B[0],a,b),a,b);}for(int e=0;e<3;++e){V q;take(point_tri_dist(A[e],B[0],B[1],B[2],q),A[e],q);take(point_tri_dist(B[e],A[0],A[1],A[2],q),q,B[e]);}return r; }

struct Query { const Mesh *a,*b; BVH *ba,*bb; const std::vector<V>*wa,*wb; Closest best; };
static void visit(Query& q,int ia,int ib) { const auto& A=q.ba->n[ia];const auto& B=q.bb->n[ib]; if(box_dist2(A.world,B.world)>=q.best.d2-1e-18&&!q.best.intersect)return; if(A.left<0&&B.left<0){for(int x:A.tris){V ta[3]={(*q.wa)[q.a->t[x].i[0]],(*q.wa)[q.a->t[x].i[1]],(*q.wa)[q.a->t[x].i[2]]};for(int y:B.tris){V tb[3]={(*q.wb)[q.b->t[y].i[0]],(*q.wb)[q.b->t[y].i[1]],(*q.wb)[q.b->t[y].i[2]]};auto r=tri_pair(ta,tb,x,y);if(r.intersect||r.d2<q.best.d2)q.best=r;if(q.best.intersect)return;}}return;} if(A.left<0){visit(q,ia,B.left);visit(q,ia,B.right);}else if(B.left<0){visit(q,A.left,ib);visit(q,A.right,ib);}else{double d1=box_dist2(q.ba->n[A.left].world,q.bb->n[B.left].world),d2=box_dist2(q.ba->n[A.left].world,q.bb->n[B.right].world),d3=box_dist2(q.ba->n[A.right].world,q.bb->n[B.left].world),d4=box_dist2(q.ba->n[A.right].world,q.bb->n[B.right].world);std::array<std::pair<double,std::pair<int,int>>,4> order={{{d1,{A.left,B.left}},{d2,{A.left,B.right}},{d3,{A.right,B.left}},{d4,{A.right,B.right}}}};std::sort(order.begin(),order.end());for(auto&o:order){visit(q,o.second.first,o.second.second);if(q.best.intersect)return;}} }

static std::vector<double> csv_fields(const std::string& s) { std::vector<double> out; std::string x=s;auto a=x.find('['),b=x.rfind(']');if(a!=std::string::npos)x=x.substr(a+1,b>a?b-a-1:x.size());std::stringstream ss(x);std::string v;while(std::getline(ss,v,','))try{out.push_back(std::stod(v));}catch(...){ }return out; }
struct Node { std::string id,pair; int wp=0,index=0; std::array<double,6> q{}; };
static std::vector<Node> read_nodes(const fs::path& p) { std::ifstream f(p);std::string line;std::getline(f,line);std::vector<Node> out;while(std::getline(f,line)){std::vector<std::string> c;std::string x;bool quote=false;for(char ch:line){if(ch=='\"')quote=!quote;else if(ch==','&&!quote){c.push_back(x);x.clear();}else x+=ch;}c.push_back(x);if(c.size()<5)continue;Node n;n.id=c[0];n.wp=std::stoi(c[1]);n.index=std::stoi(c[2]);n.pair=c[3];auto q=csv_fields(c[4]);if(q.size()!=6)continue;for(int i=0;i<6;++i)n.q[i]=q[i];out.push_back(n);}return out; }
static std::map<std::string,M> fk(const std::array<double,6>& q) { M cur=M::identity();std::map<std::string,M> out;cur=cur*transl(V{0,0,0})*rz(q[0]);out["shoulder_link"]=cur;cur=cur*transl(V{0,0,.152})*rx(1.5708)*rz(q[1]);out["upperarm_link"]=cur;cur=cur*transl(V{-.425,0,0})*rz(q[2]);out["forearm_link"]=cur;cur=cur*transl(V{-.39501,0,0})*rz(q[3]);out["wrist1_link"]=cur;cur=cur*transl(V{0,0,.1021})*rx(1.5708)*rz(q[4]);out["wrist2_link"]=cur;cur=cur*transl(V{0,0,.102})*rx(-1.5708)*rz(q[5]);out["wrist3_link"]=cur;return out; }

static std::string esc(const std::string&s){std::string o;for(char c:s){if(c=='\"'||c=='\\')o+='\\';o+=c;}return o;}
static void json_v(std::ostream&o,const V&v){o<<std::setprecision(17)<<'['<<v.x<<','<<v.y<<','<<v.z<<']';}
int main(int argc,char**argv){if(argc!=6){std::cerr<<"usage: independent_geometry_probe nodes.csv forearm.stl wrist2.stl wrist3.stl output_dir\n";return 2;}fs::path in=argv[1],pa=argv[2],pb=argv[3],pc=argv[4],od=argv[5];try{fs::create_directories(od);Mesh fore=read_stl(pa),w2=read_stl(pb),w3=read_stl(pc);BVH bf,b2,b3;bf.init(fore);b2.init(w2);b3.init(w3);auto nodes=read_nodes(in);std::ofstream out(od/"stage23a7_independent_geometry.csv");out<<"node_id,waypoint_id,pair,collision_element_a,collision_element_b,triangle_intersection,minimum_distance_m,closest_triangle_a,closest_triangle_b,closest_point_a_x,closest_point_a_y,closest_point_a_z,closest_point_b_x,closest_point_b_y,closest_point_b_z,geometry_status,transform_source\n";std::ofstream tr(od/"stage23a7_fk_transforms.csv");tr<<"node_id,waypoint_id,link,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33\n";for(const auto&n:nodes){auto T=fk(n.q);for(const auto&kv:T){const auto&t=kv.second;tr<<n.id<<','<<n.wp<<','<<kv.first;for(int i=0;i<4;++i)for(int j=0;j<4;++j)tr<<','<<std::setprecision(17)<<t.a[i][j];tr<<'\n';}std::string la=n.pair.find("wrist2")!=std::string::npos?"wrist2_link":"wrist3_link";const Mesh*ma=&fore;const Mesh*mb=la=="wrist2_link"?&w2:&w3;BVH*qa=&bf;BVH*qb=la=="wrist2_link"?&b2:&b3;std::vector<V> wa,wb;wa.reserve(ma->v.size());wb.reserve(mb->v.size());for(auto&v:ma->v)wa.push_back(apply(T["forearm_link"],v));for(auto&v:mb->v)wb.push_back(apply(T[la],v));qa->update(0,wa);qb->update(0,wb);Query query{ma,mb,qa,qb,&wa,&wb,{}};visit(query,0,0);auto&r=query.best;double d=std::sqrt(std::max(0.0,r.d2));std::string status=r.intersect?"exact_triangle_intersection":(d<=1e-8?"near_zero_separation":"strictly_positive_separation");out<<n.id<<','<<n.wp<<','<<n.pair<<",0,0,"<<(r.intersect?"true":"false")<<','<<std::setprecision(17)<<d<<','<<r.ia<<','<<r.ib<<','<<r.a.x<<','<<r.a.y<<','<<r.a.z<<','<<r.b.x<<','<<r.b.y<<','<<r.b.z<<','<<status<<",independent_fk_from_frozen_urdf_chain\n";}std::cerr<<"nodes="<<nodes.size()<<" triangles_forearm="<<fore.t.size()<<" triangles_pair2="<<w2.t.size()<<" triangles_pair3="<<w3.t.size()<<" output="<<(od/"stage23a7_independent_geometry.csv")<<"\n";return nodes.size()>=658?0:4;}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<"\n";return 3;}}
