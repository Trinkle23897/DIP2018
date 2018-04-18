#!/usr/bin/env python
import base64,json,os,argparse,cv2

def main(args):
	if args.template=='':
		print('Please specify template filename')
		return
	if args.merge=='':
		print('Please specify merge filename')
		return
	if args.rate>100:
		args.rate=100
	if args.rate<0:
		args.rate=0
	if not os.path.exists(args.template+'.json'):
		os.system('./keypoints.sh %s'%args.template)

	dic=json.loads(open(args.template+'.json').read())
	rec=dic['faces'][0]['face_rectangle']
	os.system('curl -X POST "https://api-cn.faceplusplus.com/imagepp/v1/mergeface" -F "api_key=iwJEEsmz09d0W-zo9kG8tej9HB5VAEGF" -F "api_secret=1veCRIY5N7ZWESS9qQcBpoeYOZEZAz5h" -F "template_file=@%s" -F "template_rectangle=%d,%d,%d,%d" -F "merge_file=@%s" -F "merge_rate=%d" > merge.json'%(args.template,rec['top'],rec['left'],rec['width'],rec['height'],args.merge,args.rate))
	dic=json.loads(open('merge.json').read())
	print('Used %dms'%dic['time_used'])
	with open('merge.jpg','wb') as f: 
		f.write(base64.b64decode(dic['result']))
	if args.output:
		a=cv2.imread('merge.jpg')
		cv2.imwrite(args.output,a)
		os.system('rm merge.jpg')
		print('Output result in %s'%args.output)
	else:
		print('Output result in merge.jpg')

if __name__=='__main__':
	parser=argparse.ArgumentParser(description='Merge two faces through Face++ API')
	parser.add_argument('-t', dest='template', type=str, help='Template image filename')
	parser.add_argument('-m', dest='merge', type=str, help='Merge image filename')
	parser.add_argument('-o', dest='output', type=str, help='Output image filename ')
	parser.add_argument('-r', dest='rate', default=50, type=int, help='merge rate (0~100), 0: template; 100: merge')
	args=parser.parse_args()
	main(args)