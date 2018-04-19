#define STB_IMAGE_IMPLEMENTATION
#include "tools/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tools/stb_image_write.h"
#include "delaunay.h"

namespace Gauss
{
ld a[10][10];
const int n=3;
void pr(){
	for(int i=1;i<=n;i++,puts(""))
		for(int j=1;j<=n+1;j++)
			printf("%lf ",a[i][j]);
	puts("");
}
void solve(){
	int i,j,k,las;ld t;
	for(i=1;i<=n;i++) {
		for(t=0,las=j=i;j<=n;j++)
			if(abs(a[j][i])>t)t=abs(a[j][i]),las=j;
		if(j=las,j!=i)
			for(k=1;k<=n+1;k++)t=a[i][k],a[i][k]=a[j][k],a[j][k]=t;
		for(j=i+1;j<=n;j++)
			for(t=a[j][i]/a[i][i],k=i;k<=n+1;k++)a[j][k]-=a[i][k]*t;
	}
	for(i=n;i>=1;i--)
		for(a[i][n+1]/=a[i][i],j=i-1;j;j--)a[j][n+1]-=a[j][i]*a[i][n+1];
}
}
struct Mat{
	ld m[2][3];
	P transform(PP p){return(P){p.x*m[0][0]+p.y*m[0][1]+m[0][2],p.x*m[1][0]+p.y*m[1][1]+m[1][2]};}
}f1,f2;
void Calc_Transform_Matrix(Mat&f,P a1,P b1,P c1,P a2,P b2,P c2)
{
	/*
	 * | a b c |   | 1x |   | 2x |
	 * | d e f | * | 1y | = | 2y |
	 * | 0 0 1 |   | 1  |   | 1  |
	 *
	 * solve a,b,c,d,e,f -> Matrix f.m
	 */
	using namespace Gauss;
	//a,b,c
	a[1][1]=a1.x,a[2][1]=b1.x,a[3][1]=c1.x;
	a[1][2]=a1.y,a[2][2]=b1.y,a[3][2]=c1.y;
	a[1][3]=a[2][3]=a[3][3]=1;
	a[1][4]=a2.x,a[2][4]=b2.x,a[3][4]=c2.x;
	solve();
	f.m[0][0]=a[1][4],f.m[0][1]=a[2][4],f.m[0][2]=a[3][4];
	//d,e,f
	a[1][1]=a1.x,a[2][1]=b1.x,a[3][1]=c1.x;
	a[1][2]=a1.y,a[2][2]=b1.y,a[3][2]=c1.y;
	a[1][3]=a[2][3]=a[3][3]=1;
	a[1][4]=a2.y,a[2][4]=b2.y,a[3][4]=c2.y;
	solve();
	f.m[1][0]=a[1][4],f.m[1][1]=a[2][4],f.m[1][2]=a[3][4];
}

#define getidx(i,j,k,w,c) ((i)*(w)*(c)+(j)*(c)+(k))
struct IMG{
	ld*img;
	unsigned char*buf;
	std::string filename;
	P pt[120];
	int w,h,c,len;
	IMG():img(NULL),w(0),h(0),c(0),len(0){}
	IMG(int _h,int _w,int _c=1)
	{
		len=0;
		init(_h,_w,_c);
	}
	void init(int _h,int _w,int _c)
	{
		h=_h,w=_w,c=_c;
		img=new ld[w*h*c];
		buf=new unsigned char[w*h*c];
		memset(img,0,sizeof(ld)*w*h*c);
		memset(buf,0,sizeof(unsigned char)*w*h*c);
	}
	IMG(std::string _):filename(_){
		if(_=="")return;
		buf=stbi_load(filename.c_str(),&w,&h,&c,0);
		printf("load image w=%d h=%d c=%d\n",w,h,c);
		std::string txtname=_+".txt";
		FILE*fin=fopen(txtname.c_str(),"r");
		fscanf(fin,"%*d%*d%*d%*d");
		int x,y;len=0;
		while(~fscanf(fin,"%d%d",&x,&y))
			pt[++len]=(P){x,y};
		pt[++len]=(P){0,0};
		pt[++len]=(P){0,w/2};
		pt[++len]=(P){0,w-1};
		pt[++len]=(P){h/2,w-1};
		pt[++len]=(P){h-1,w-1};
		pt[++len]=(P){h-1,w/2};
		pt[++len]=(P){h-1,0};
		pt[++len]=(P){h/2,0};
		// for(int i=1;i<=len;++i)
		// {
			// int x=int(pt[i].x+.5),y=int(pt[i].y+.5);
			// std::swap(pt[i].x,pt[i].y);
			// printf("%d %d\n",x,y);
			// buf[getidx(x,y,0,w,c)]=buf[getidx(x,y,1,w,c)]=buf[getidx(x,y,2,w,c)]=255;
		// }
		img=new ld[w*h*c];
		for(int i=0;i<w*h*c;++i)
			img[i]=buf[i];
		DT::solve(len,pt);
	}
	void write(const char* output_filename){
		buf=new unsigned char[w*h*c];
		for(int i=0;i<w*h*c;++i)
			buf[i]=img[i]<0?0:img[i]>255?255:img[i];
		stbi_write_png(output_filename,w,h,c,buf,0);
	}
	void print(char*s)
	{
		puts(s);
		puts("--u");
		for(int i=0;i<w;++i,puts(""))
			for(int j=0;j<h;++j)
				printf("%6d ",1*buf[getidx(i,j,0,w,c)]);
				// printf("%d,%d,%d ",1l*buf[getidx(i,j,0,w,c)],1l*buf[getidx(i,j,1,w,c)],1l*buf[getidx(i,j,2,w,c)]);
		puts("--f");
		for(int i=0;i<w;++i,puts(""))
			for(int j=0;j<h;++j)
				printf("%7.1f",img[getidx(i,j,0,w,c)]);
				// printf("%.1lf,%.1lf,%.1lf ",img[getidx(i,j,0,w,c)],img[getidx(i,j,1,w,c)],img[getidx(i,j,2,w,c)]);
		puts("");
	}
};

#define getpix(f,i,j,k) f.img[getidx(i,j,k,f.w,f.c)]
#define getbuf(f,i,j,k) f.buf[getidx(i,j,k,f.w,f.c)]
bool intri(PP p,PP a,PP b,PP c)
{
	ld s=std::abs(check(a,b,c));
	ld s_=std::abs(check(a,p,b))+std::abs(check(b,p,c))+std::abs(check(c,p,a));
	return std::abs(s-s_)<1e-3;
}
int main(int argc, char *argv[])
{
	std::string fn1=argv[1],fn2=argv[2],output_filename=argv[3];
	ld rate=atoi(argv[4])/100.;
	IMG img1(fn1),img2(fn2);
	IMG img3(img1.h,img1.w,3);

	assert(img1.c==img2.c);
	for(int i=1;i<=DT::tri_cnt;++i)
	{
		int x=DT::triangle[i].x,y=DT::triangle[i].y,z=DT::triangle[i].z;
		P a1=img1.pt[x],b1=img1.pt[y],c1=img1.pt[z];
		P a2=img2.pt[x],b2=img2.pt[y],c2=img2.pt[z];
		P a3=a1*(1-rate)+a2*rate,b3=b1*(1-rate)+b2*rate,c3=c1*(1-rate)+c2*rate;
		Calc_Transform_Matrix(f1,a3,b3,c3,a1,b1,c1);
		Calc_Transform_Matrix(f2,a3,b3,c3,a2,b2,c2);
		int min_x=std::min(std::min(a3.x,b3.x),c3.x);
		int max_x=std::max(std::max(a3.x,b3.x),c3.x)+1;
		int min_y=std::min(std::min(a3.y,b3.y),c3.y);
		int max_y=std::max(std::max(a3.y,b3.y),c3.y)+1;
		// printf("%f %f %f %f %f %f\n",a1.x,a1.y,b1.x,b1.y,c1.x,c1.y);
		// printf("%f %f %f %f %f %f\n",a2.x,a2.y,b2.x,b2.y,c2.x,c2.y);
		// printf("%f %f %f %f %f %f\n",a3.x,a3.y,b3.x,b3.y,c3.x,c3.y);puts("");
		for(int i=min_x;i<=max_x;++i)
			for(int j=min_y;j<=max_y;++j)
				if(intri((P){i,j},a3,b3,c3))
				{
					// printf("fill %d %d\n",i,j);
					P tar1=f1.transform((P){i,j}),tar2=f2.transform((P){i,j});
					int x1=int(tar1.x+.5),y1=int(tar1.y+.5);
					int x2=int(tar2.x+.5),y2=int(tar2.y+.5);
					// printf("trans1 %d %d   trans2 %d %d\n",x1,y1,x2,y2);
					for(int k=0;k<img3.c&&k<3;++k)
						getpix(img3,i,j,k)=(1-rate)*getpix(img1,x1,y1,k)+rate*getpix(img2,x2,y2,k);//,printf("%f ",getpix(img3,i,j,k));
					// puts("");
				}
	}
	img3.write(output_filename.c_str());

	return 0;
}