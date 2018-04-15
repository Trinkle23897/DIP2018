#include <bits/stdc++.h>
#define STB_IMAGE_IMPLEMENTATION
#include "tools/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tools/stb_image_write.h"
#define ld float
int dx[]={-1,0,0,1},dy[]={0,-1,1,0};
char str[100];
#define getidx(i,j,k,w,c) ((i)*(w)*(c)+(j)*(c)+(k))
struct IMG{
	ld*img;
	unsigned char*buf;
	std::string filename;
	int w,h,c,w0,w1,h0,h1;
	IMG():img(NULL),w(0),h(0),c(0){}
	IMG(int _h,int _w,int _c=1):h(_h),w(_w),c(_c)
	{
		img=new ld[w*h*c];
		buf=new unsigned char[w*h*c];
		memset(img,0,sizeof(ld)*w*h*c);
		memset(buf,0,sizeof(unsigned char)*w*h*c);
		w0=h0=1<<30,w1=h1=1;
	}
	IMG(std::string _):filename(_){
		buf=stbi_load(filename.c_str(),&w,&h,&c,0);
		img=new ld[w*h*c];
		for(int i=0;i<w*h*c;++i)
			img[i]=buf[i];
		stat();
	}
	void stat(int lb=0){
		h0=w0=1<<30,h1=w1=0;
			for(int i=0;i<h;++i)
				for(int j=0;j<w;++j)
					for(int k=0;k<c;++k)
						buf[getidx(i,j,k,w,c)]>lb?i<h0?h0=i:1,h1<i?h1=i:1,j<w0?w0=j:1,w1<j?w1=j:1:1;
		// printf("h: %d,%d; w: %d,%d\n",h0,h1,w0,w1);
	}
	void write(char* output_filename){
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
				printf("%6d ",1l*buf[getidx(i,j,0,w,c)]);
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

struct Solver{
// 4x1-x2-x3=4
// var var const
	int label,id,size;
	ld *x,*b,*tmp;int **a; // a[id][0]: number; a[id][>0]: -id
	void resize(int _size)
	{
		x=new ld[size=_size];memset(x,0,sizeof(ld)*size);
		tmp=new ld[size];memset(tmp,0,sizeof(ld)*size);
		a=new int*[size];
		b=new ld[size];memset(b,0,sizeof(ld)*size);
		for(int i=0;i<size;++i)
		{
			a[i]=new int[10];
			a[i][0]=0;
		}
	}
	void additem(int _id,int _x=0){
		// printf("size=%d id=%d\n",size,_id);
		x[id=_id]=_x;
		b[id]=a[id][0]=0;
		// x[id]=0;
	}
	void addconst(int _b){b[id]+=_b;}
	void addvar(int id2){a[id][++a[id][0]]=id2;}
	void print(){
		printf("size=%d\n",size);
		// return;
		for(int i=0;i<size;++i)
			if(a[i][0]){
				printf("4x%d",i);
				for(int j=1;j<=a[i][0];++j)
					printf("-x%d",a[i][j]);
				printf("=%.1lf\n",b[i]);
			}
	}
	void iter(){
		// calc tmp=4x+b-Ax
		for(int i=0;i<size;++i)
			if(a[i][0])
			{
				tmp[i]=b[i];//-4*x[i];
				for(int j=1;j<=a[i][0];++j)
					tmp[i]+=x[a[i][j]];
			}
		for(int i=0;i<size;++i)
			if(a[i][0])
				x[i]=tmp[i]/4;
	}
};

int main(int argc, char const *argv[])
{
	// read data
	IMG src("test1_src.jpg"),mask("test1_mask.jpg"),target("test1_target.jpg");
	// src.print("src");
	// mask.print("mask");
	// target.print("tar");
	int ph=50,pw=100;
	assert(src.c==mask.c);
	assert(src.c==target.c);
	printf("src size: %d*%d\n",src.h,src.w);
	mask.stat(200);
	printf("mask size: %d*%d [%d,%d]*[%d,%d]\n",mask.h,mask.w,mask.h0,mask.h1,mask.w0,mask.w1);
	printf("target size: %d*%d\n",target.h,target.w);
	// init
	IMG src_grad(mask.h1-mask.h0+1,mask.w1-mask.w0+1,mask.c);
	// src_grad.print("grad");

	for(int i=mask.h0,h=0;i<=mask.h1;++i,++h)
		for(int j=mask.w0,w=0;j<=mask.w1;++j,++w)
			for(int k=0;k<=mask.c;++k)
				if(getpix(mask,i,j,k)>200)
					getbuf(src_grad,h,w,k)=255,
					getpix(src_grad,h,w,k)=getpix(src,i,j,k)*4
										  -getpix(src,i-1,j,k)
										  -getpix(src,i+1,j,k)
										  -getpix(src,i,j-1,k)
										  -getpix(src,i,j+1,k);
	src_grad.stat(200);
	// src_grad.print("grad");
	printf("src_grad h:[%d,%d] w:[%d,%d]\n",src_grad.h0,src_grad.h1,src_grad.w0,src_grad.w1);
	printf("grad done\n");
	// init solver
	Solver*sv=new Solver[mask.c];
	for(int k=0;k<mask.c;++k)
	{
		sv[k].label=k;
		sv[k].resize((src_grad.h1-src_grad.h0+1)*src_grad.w+src_grad.w1-src_grad.w0+1);
	}
	printf("solver init\n");
	for(int i=src_grad.h0,h=ph;i<=src_grad.h1;++i,++h)
		for(int j=src_grad.w0,w=pw;j<=src_grad.w1;++j,++w)
			// (i,j) -> src_grad
			// (h,w) -> target
			for(int k=0;k<mask.c;++k)
				if(getbuf(src_grad,i,j,k)==255)
				{
					// printf("i=%d j=%d k=%d\n",i,j,k);
					// printf("mapping: %d,%d\n",i-src_grad.h0+mask.h0,j-src_grad.w0+mask.w0,k);
					sv[k].additem(i*src_grad.w+j,getpix(src,i-src_grad.h0+mask.h0,j-src_grad.w0+mask.w0,k));
					sv[k].addconst(getpix(src_grad,i,j,k));
					if(getbuf(src_grad,i-1,j,k)==255)
						sv[k].addvar((i-1)*src_grad.w+j);
					else
						sv[k].addconst(getpix(target,h-1,w,k));
					if(getbuf(src_grad,i+1,j,k)==255)
						sv[k].addvar((i+1)*src_grad.w+j);
					else
						sv[k].addconst(getpix(target,h+1,w,k));
					if(getbuf(src_grad,i,j-1,k)==255)
						sv[k].addvar(i*src_grad.w+j-1);
					else
						sv[k].addconst(getpix(target,h,w-1,k));
					if(getbuf(src_grad,i,j+1,k)==255)
						sv[k].addvar(i*src_grad.w+j+1);
					else
						sv[k].addconst(getpix(target,h,w+1,k));
				}
	printf("solver init done\n");
	// sv[0].print();//return 0;
	// iterate solver
	for(int _=0;_<=2000;_++)
	{
		int vali=0;
		for(int i=src_grad.h0,h=ph;i<=src_grad.h1;++i,++h)
			for(int j=src_grad.w0,w=pw;j<=src_grad.w1;++j,++w)
				for(int k=0;k<mask.c;++k)
					if(getbuf(src_grad,i,j,k)==255)
						getpix(target,h,w,k)=sv[k].x[vali=i*src_grad.w+j];
		if(_%20==0||_<10)
		{
			printf("%d %lf\n",_,sv[2].x[vali]);
			sprintf(str,"iter%d.png",_);
			target.write(str);
		}
		for(int k=0;k<mask.c;++k)
			sv[k].iter();
	}
}
