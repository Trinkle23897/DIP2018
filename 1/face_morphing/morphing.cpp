#define STB_IMAGE_IMPLEMENTATION
#include "tools/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tools/stb_image_write.h"
#include "delaunay.h"

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
		int x,y,mx=0,my=0;len=0;
		while(~fscanf(fin,"%d%d",&x,&y))
			pt[++len]=(P){x,y},mx<x?mx=x:1,my<y?my=y:1;
		pt[++len]=(P){0,0};
		pt[++len]=(P){0,w/2};
		pt[++len]=(P){0,w-1};
		pt[++len]=(P){h/2,w-1};
		pt[++len]=(P){h-1,w-1};
		pt[++len]=(P){h-1,w/2};
		pt[++len]=(P){h-1,0};
		pt[++len]=(P){h/2,0};
		for(int i=1;i<=len;++i)
		{
			int x=pt[i].x,y=pt[i].y;
			buf[getidx(x,y,0,w,c)]=buf[getidx(x,y,1,w,c)]=buf[getidx(x,y,2,w,c)]=255;
		}
		img=new ld[w*h*c];
		for(int i=0;i<w*h*c;++i)
			img[i]=buf[i];
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

int main(int argc, char *argv[])
{
	std::string fn1=argv[1],fn2=argv[2],output_filename=argv[3];
	int rate=atoi(argv[4]);
	IMG img1(fn1),img2(fn2);
	assert(img1.c==img2.c);
	img1.write(output_filename.c_str());
	return 0;
}