#include <bits/stdc++.h>
#define STB_IMAGE_IMPLEMENTATION
#include "tools/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tools/stb_image_write.h"
#define ld float
int dx[]={-1,0,0,1},dy[]={0,-1,1,0};
#define getidx(i,j,k,w,c) (i*w*c+j*c+k)
struct IMG{
	ld*img;
	unsigned char*buf;
	std::string filename;
	int w,h,c,w0,w1,h0,h1;
	IMG():img(NULL),w(0),h(0),c(0){}
	IMG(int _h,int _w,int _c=1):h(_h),w(_w),c(_c){img=new ld[w*h*c];w0=h0=0,w1=h1=1;}
	IMG(std::string _):filename(_){
		buf=stbi_load(filename.c_str(),&w,&h,&c,0);
		img=new ld[w*h*c];
		for(int i=0;i<w*h*c;++i)
			img[i]=buf[i];
		stat();
	}
	void stat(){
		h0=w0=1<<30,h1=w1=0;
			for(int i=0;i<h;++i)
				for(int j=0;j<w;++j)
					for(int k=0;k<c;++k)
						buf[getidx(i,j,k,w,c)]?i<h0?h0=i:1,h1<i?h1=i:1,j<w0?w0=j:1,w1<j?w1=j:1:1;
	}
	void write(std::string output_filename){
		buf=new unsigned char[w*h*c];
		for(int i=0;i<w*h*c;++i)
			buf[i]=img[i];
		stbi_write_png(output_filename.c_str(),w,h,c,buf,0);
	}
};

#define getpix(f,i,j,k) f.img[getidx(i,j,k,f.w,f.c)]
#define getbuf(f,i,j,k) f.buf[getidx(i,j,k,f.w,f.c)]

struct Solver{
	int label;
// x1-x3=4
// var var const
};

int main(int argc, char const *argv[])
{
	// read data
	IMG src("test1_src.jpg"),mask("test1_mask.jpg"),target("test1_target.jpg");
	int ph=50,pw=100;
	assert(src.c==mask.c);
	assert(src.c==target.c);
	printf("src size: %d*%d\n",src.h,src.w);
	printf("mask size: %d*%d [%d,%d]*[%d,%d]\n",mask.h,mask.w,mask.h0,mask.h1,mask.w0,mask.w1);
	printf("target size: %d*%d\n",target.h,target.w);
	// init
	IMG src_grad(mask.h1-mask.h0+1,mask.w1-mask.w0+1,mask.c);

	for(int i=mask.h0+1,h=1;i<mask.h1;++i,++h)
		for(int j=mask.w0+1,w=1;j<mask.w1;++j,++w)
			for(int k=0;k<=mask.c;++k)
				if(getpix(src,i-1,j,k)>200&&getpix(src,i+1,j,k)>200
				 &&getpix(src,i,j-1,k)>200&&getpix(src,i,j+1,k)>200)
					getbuf(src_grad,h,w,k)=255,
					getpix(src_grad,h,w,k)=getpix(src,i,j,k)*4
										  -getpix(src,i-1,j,k)
										  -getpix(src,i+1,j,k)
										  -getpix(src,i,j-1,k)
										  -getpix(src,i,j+1,k);
	src_grad.stat();
	// init solver
	Solver*sv=new Solver[mask.c];
	for(int k=0;k<mask.c;++k)
		sv[k].label=k;
	for(int i=src_grad.h0,h=ph;i<=src_grad.h1;++i,++h)
		for(int j=src_grad.w0,w=pw;j<=src_grad.w1;++j,++w)
			// (i,j) -> src_grad
			// (h,w) -> target
			for(int k=0;k<mask.c;++k)
				if(getbuf(src_grad,i,j,k))
				{
					sv[k].additem(i*src_grad.w+j);
					sv[k].addconst(getpix(src_grad,i,j,k),0);
					if(getbuf(src_grad,i-1,j,k)==255)
						sv[k].addvar((i-1)*src_grad.w+j);
					else
						sv[k].addconst(getpix(target,h-1,w,k),1);
					if(getbuf(src_grad,i+1,j,k)==255)
						sv[k].addvar((i+1)*src_grad.w+j);
					else
						sv[k].addconst(getpix(target,h+1,w,k),1);
					if(getbuf(src_grad,i,j-1,k)==255)
						sv[k].addvar(i*src_grad.w+j-1);
					else
						sv[k].addconst(getpix(target,h,w-1,k),1);
					if(getbuf(src_grad,i,j+1,k)==255)
						sv[k].addvar(i*src_grad.w+j+1);
					else
						sv[k].addconst(getpix(target,h,w+1,k),1);
				}
	// iterate solver
	for(int _=0;_<20;_++)
	{
		// r
		// g
		// b
		// debug
	}
	//save
}
