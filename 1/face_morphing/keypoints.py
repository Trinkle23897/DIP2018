#!/usr/bin/env python
import os,json,argparse
from collections import OrderedDict
if __name__=='__main__':
	parser=argparse.ArgumentParser(description='Get 106 facial keypoints through Face++ API')
	parser.add_argument('-f', dest='file', type=str, help='Image filename')
	args=parser.parse_args()
	if args.file:
		if not os.path.exists(args.file+'.json'):
			os.system('curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" -F "api_key=iwJEEsmz09d0W-zo9kG8tej9HB5VAEGF" -F "api_secret=1veCRIY5N7ZWESS9qQcBpoeYOZEZAz5h" -F "image_file=@%s" -F "return_landmark=2" > %s.json'%(args.file,args.file))
		dic=OrderedDict(json.loads(open(args.file+'.json').read()))
		with open(args.file+'.txt','w') as f:
			rec=dic['faces'][0]['face_rectangle']
			f.write('%d %d %d %d\n'%(rec['top'],rec['left'],rec['width'],rec['height']))
			for i in dic['faces'][0]['landmark']:
				tmp=dic['faces'][0]['landmark'][i]
				f.write('%d %d\n'%(tmp['y'],tmp['x']))
	else:
		print('Hint: ./keypoints -h')
