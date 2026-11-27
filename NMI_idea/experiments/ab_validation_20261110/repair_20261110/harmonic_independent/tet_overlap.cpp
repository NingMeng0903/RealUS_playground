#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <vector>
using namespace std;
struct V{double x,y,z;V operator-(const V&b)const{return{x-b.x,y-b.y,z-b.z};}};
double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}V cross(V a,V b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
struct B{double lo[3],hi[3];};struct Node{B b;int l=-1,r=-1;uint32_t begin,end;};
vector<V>v;vector<array<uint32_t,4>>t;vector<int32_t>layer;vector<B>boxes;vector<uint32_t>ids;vector<Node>tree;
constexpr double eps=1e-10;uint64_t candidate=0,satcalls=0,overlap=0,shared=0,disjoint=0,crosslayer=0,distantlayer=0;priority_queue<array<double,6>>examples;
double coord(V a,int j){return j==0?a.x:j==1?a.y:a.z;}
bool meet(const B&a,const B&b){for(int j=0;j<3;j++)if(min(a.hi[j],b.hi[j])-max(a.lo[j],b.lo[j])<=eps)return false;return true;}
B rangebox(uint32_t a,uint32_t b){B box;for(int j=0;j<3;j++){box.lo[j]=1e300;box.hi[j]=-1e300;}for(auto k=a;k<b;k++)for(int j=0;j<3;j++){box.lo[j]=min(box.lo[j],boxes[ids[k]].lo[j]);box.hi[j]=max(box.hi[j],boxes[ids[k]].hi[j]);}return box;}
int build(uint32_t a,uint32_t b){int n=tree.size();tree.push_back({rangebox(a,b),-1,-1,a,b});if(b-a<=8)return n;int ax=0;for(int j=1;j<3;j++)if(tree[n].b.hi[j]-tree[n].b.lo[j]>tree[n].b.hi[ax]-tree[n].b.lo[ax])ax=j;auto mid=(a+b)/2;nth_element(ids.begin()+a,ids.begin()+mid,ids.begin()+b,[ax](auto i,auto j){return boxes[i].lo[ax]+boxes[i].hi[ax]<boxes[j].lo[ax]+boxes[j].hi[ax];});int l=build(a,mid),r=build(mid,b);tree[n].l=l;tree[n].r=r;return n;}
bool axis(V a,const array<V,4>&p,const array<V,4>&q,double&margin){double norm=sqrt(dot(a,a));if(norm<1e-20)return true;double pl=1e300,ph=-1e300,ql=1e300,qh=-1e300;for(int k=0;k<4;k++){double x=dot(a,p[k])/norm,y=dot(a,q[k])/norm;pl=min(pl,x);ph=max(ph,x);ql=min(ql,y);qh=max(qh,y);}double d=min(ph,qh)-max(pl,ql);margin=min(margin,d);return d>eps;}
bool sat(uint32_t i,uint32_t j,double&margin){array<V,4>p,q;for(int k=0;k<4;k++){p[k]=v[t[i][k]];q[k]=v[t[j][k]];}int fs[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};for(auto&f:fs){if(!axis(cross(p[f[1]]-p[f[0]],p[f[2]]-p[f[0]]),p,q,margin))return false;if(!axis(cross(q[f[1]]-q[f[0]],q[f[2]]-q[f[0]]),p,q,margin))return false;}int es[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};for(auto&e:es)for(auto&f:es)if(!axis(cross(p[e[1]]-p[e[0]],q[f[1]]-q[f[0]]),p,q,margin))return false;return true;}
void pairtest(uint32_t i,uint32_t j){candidate++;if(!meet(boxes[i],boxes[j]))return;satcalls++;double margin=1e300;if(!sat(i,j,margin))return;overlap++;int ns=0;for(auto a:t[i])for(auto b:t[j])ns+=a==b;if(ns)shared++;else disjoint++;if(layer[i]!=layer[j])crosslayer++;if(abs(layer[i]-layer[j])>=2)distantlayer++;array<double,6> rec={double(min(i,j)),double(max(i,j)),double(ns),margin,double(layer[i]),double(layer[j])};if(examples.size()<1024)examples.push(rec);else if(rec<examples.top()){examples.pop();examples.push(rec);}}
void walk(int a,int b){auto&A=tree[a];auto&B=tree[b];if(!meet(A.b,B.b))return;if(a==b){if(A.l<0){for(auto i=A.begin;i<A.end;i++)for(auto j=i+1;j<A.end;j++)pairtest(ids[i],ids[j]);}else{walk(A.l,A.l);walk(A.r,A.r);walk(A.l,A.r);}return;}if(A.l<0&&B.l<0){for(auto i=A.begin;i<A.end;i++)for(auto j=B.begin;j<B.end;j++)pairtest(ids[i],ids[j]);return;}if(B.l<0||(A.l>=0&&(A.end-A.begin>=B.end-B.begin))){walk(A.l,b);walk(A.r,b);}else{walk(a,B.l);walk(a,B.r);}}
int main(int argc,char**argv){if(argc!=2)return 2;ifstream f(argv[1],ios::binary);uint64_t nv,nt;f.read((char*)&nv,8);f.read((char*)&nt,8);v.resize(nv);t.resize(nt);layer.resize(nt);f.read((char*)v.data(),nv*sizeof(V));f.read((char*)t.data(),nt*sizeof(t[0]));f.read((char*)layer.data(),nt*sizeof(layer[0]));if(!f)return 3;auto started=chrono::steady_clock::now();boxes.resize(nt);ids.resize(nt);iota(ids.begin(),ids.end(),0);for(uint32_t i=0;i<nt;i++)for(int j=0;j<3;j++){boxes[i].lo[j]=1e300;boxes[i].hi[j]=-1e300;for(auto k:t[i]){boxes[i].lo[j]=min(boxes[i].lo[j],coord(v[k],j));boxes[i].hi[j]=max(boxes[i].hi[j],coord(v[k],j));}}tree.reserve(nt/3);build(0,nt);walk(0,0);cout.precision(17);cout<<"{\"complete\":true,\"tet_count\":"<<nt<<",\"bvh_nodes\":"<<tree.size()<<",\"leaf_candidate_pairs\":"<<candidate<<",\"strict_aabb_pairs\":"<<satcalls<<",\"positive_interior_overlap_pairs\":"<<overlap<<",\"shared_vertex_overlap_pairs\":"<<shared<<",\"disjoint_vertex_overlap_pairs\":"<<disjoint<<",\"cross_station_bin_overlap_pairs\":"<<crosslayer<<",\"distant_station_bin_overlap_pairs\":"<<distantlayer<<",\"tolerance_m\":"<<eps<<",\"seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-started).count()<<",\"examples\":[";vector<array<double,6>>rows;while(!examples.empty()){rows.push_back(examples.top());examples.pop();}sort(rows.begin(),rows.end());for(size_t i=0;i<rows.size();i++){if(i)cout<<",";cout<<"[";for(int j=0;j<6;j++){if(j)cout<<",";cout<<rows[i][j];}cout<<"]";}cout<<"]}\n";}
