#!/usr/bin/env python
import os,cv2,argparse

def main(args):
	# if not os.path.exists('morphing'):
	print('generate bin file ...')
	os.system('g++ morphing.cpp -o morphing -O2')
	if not args.template or args.template=='':
		print('Please specify template filename, using -t option')
		return
	if not args.merge or args.merge=='':
		print('Please specify merge filename, using -m option')
		return
	if not args.output or args.output=='':
		args.output='merge.png'
	if args.rate>100:
		args.rate=100
	if args.rate<0:
		args.rate=0
	if not os.path.exists(args.template+'.txt'):
		os.system('./keypoints.py -f %s'%args.template)
	if not os.path.exists(args.merge+'.txt'):
		os.system('./keypoints.py -f %s'%args.merge)
	os.system('./morphing %s %s %s %d'%(args.template,args.merge,args.output,args.rate))
	print('Output result image in %s'%args.output)
	if args.output!='merge.png':
		cv2.imwrite(args.output,cv2.imread('merge.png'))
		os.system('rm merge.png')

if __name__=='__main__':
	parser=argparse.ArgumentParser(description='Merge two faces with traditional implements')
	parser.add_argument('-t', dest='template', type=str, help='Template image filename')
	parser.add_argument('-m', dest='merge', type=str, help='Merge image filename')
	parser.add_argument('-o', dest='output', type=str, help='Output image filename ')
	parser.add_argument('-r', dest='rate', default=50, type=int, help='merge rate (0~100), 0: template; 100: merge')
	args=parser.parse_args()
	main(args)
