#include <bits/stdc++.h>
#define STB_IMAGE_IMPLEMENTATION
#include "tools/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tools/stb_image_write.h"
unsigned char *img=NULL;
int w,h,c;
int main()
{
    img=stbi_load("test1_src.jpg",&w,&h,&c,0);
    printf("%d %d %d\n",w,h,c);
    printf("%d\n",sizeof img);
    printf("%d %d %d\n",img[267*3-3],img[267*3-2],img[267*3-1]);
    stbi_write_png("try.png",w,h,c,img,0);
}
