#include<bits/stdc++.h>
#define N 310
typedef float ld;
struct P{
	ld x,y;
#define PP register const P&
	bool operator<(PP a)const {return x<a.x||x==a.x&&y<a.y;}
	P operator-(PP a)const {return (P){x-a.x,y-a.y};}
	ld operator&(PP a)const {return x*a.y-y*a.x;}
	ld operator|(PP a)const {return x*a.x+y*a.y;}
}p[N],around[N];
#define check(a,b,c) ((b-a)&(c-a))
ld dis2(PP a){return a.x*a.x+a.y*a.y;}
#define cross(a,b,c,d) (check(p[a],p[c],p[d])*check(p[b],p[c],p[d])<0&&check(p[c],p[a],p[b])*check(p[d],p[a],p[b])<0)
namespace DT{
struct P3{
	ld x,y,z;
	bool operator<(const P3&a)const {return x<a.x||x==a.x&&(y<a.y||y==a.y&&z<a.z);}
	P3 operator-(const P3&a)const {return (P3){x-a.x,y-a.y,z-a.z};}
	ld operator|(const P3&a)const {return x*a.x+y*a.y+z*a.z;}
	P3 operator&(const P3&a)const {return (P3){y*a.z-z*a.y,z*a.x-x*a.z,x*a.y-y*a.x};}
}ori[N],tri[N];
#define gp3(a) (P3){a.x,a.y,a.x*a.x+a.y*a.y}
bool incir(int a,int b,int c,int d){
	P3 aa=gp3(p[a]),bb=gp3(p[b]),cc=gp3(p[c]),dd=gp3(p[d]);
	if(check(p[a],p[b],p[c])<0)std::swap(bb,cc);
	return (check(aa,bb,cc)|(dd-aa))<0;
}
int et=1,la[N],tot,q[N<<2],map[N][N];
struct E{int to,l,r;}e[N<<5];
void add(int x,int y){
	e[++et]=(E){y,la[x]},e[la[x]].r=et,la[x]=et;
	e[++et]=(E){x,la[y]},e[la[y]].r=et,la[y]=et;
}
void del(int x){
	e[e[x].r].l=e[x].l,e[e[x].l].r=e[x].r,la[e[x^1].to]==x?la[e[x^1].to]=e[x].l:1;
}
void delaunay(int l,int r){
	if(r-l<=2){
		for(int i=l;i<r;i++)
		for(int j=i+1;j<=r;j++)add(i,j);
		return;
	}
	int i,j,mid=l+r>>1,ld=0,rd=0,id,op;
	delaunay(l,mid),delaunay(mid+1,r);
	for(tot=0,i=l;i<=r;q[++tot]=i++)
	while(tot>1&&check(p[q[tot-1]],p[q[tot]],p[i])<0)tot--;
	for(i=1;i<tot&&!ld;i++)if(q[i]<=mid&&mid<q[i+1])ld=q[i],rd=q[i+1];
	for(;add(ld,rd),1;){
		id=op=0;
		for(i=la[ld];i;i=e[i].l)if(check(p[ld],p[rd],p[e[i].to])>0)
		if(!id||incir(ld,rd,id,e[i].to))op=-1,id=e[i].to;
		for(i=la[rd];i;i=e[i].l)if(check(p[rd],p[ld],p[e[i].to])<0)
		if(!id||incir(ld,rd,id,e[i].to))op=1,id=e[i].to;
		if(op==0)break;
		if(op==-1){
			for(i=la[ld];i;i=e[i].l)
			if(cross(rd,id,ld,e[i].to))del(i),del(i^1),i=e[i].r;
			ld=id;
		}else{
			for(i=la[rd];i;i=e[i].l)
			if(cross(ld,id,rd,e[i].to))del(i),del(i^1),i=e[i].r;
			rd=id;
		}
	}
}
struct Tri{
	int x,y,z;//id
	void sort(){if(x>y)std::swap(x,y);if(x>z)std::swap(x,z);if(y>z)std::swap(y,z);}
	bool operator<(const Tri&a)const {return x<a.x||x==a.x&&(y<a.y||y==a.y&&z<a.z);}
	bool operator==(const Tri&a)const {return x==a.x&&y==a.y&&z==a.z;}
}triangle[N];

void solve(int n,P*_,int&tri_cnt){
	for(int i=1;i<=n;i++)p[i]=_[i],ori[i]=(P3){p[i].x,p[i].y,i};
	std::sort(p+1,p+1+n);std::sort(ori+1,ori+1+n);delaunay(1,n);	
	memset(map,0,sizeof map);
	for(int i=1;i<=n;i++)
		for(int x=ori[i].z,y,j=la[i];j;j=e[j].l){
			y=ori[e[j].to].z,map[x][y]=map[y][x]=1;
		}
	tri_cnt=0;
	for(int i=1;i<=n;++i)
		for(int j=i+1;j<=n;++j)
			if(map[i][j])
				for(int k=j+1;k<=n;++k)
					if(map[i][k]&&map[j][k]&&std::abs(check(_[i],_[j],_[k]))>1e-3)
						triangle[++tri_cnt]=(Tri){i,j,k};
	std::sort(triangle+1,triangle+1+tri_cnt);
}
}