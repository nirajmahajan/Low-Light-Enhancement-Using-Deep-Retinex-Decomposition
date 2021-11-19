import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type = int, default = '0')
args = parser.parse_args()

task_list = ['22','23', '24', '25', '26']

prev_task = None
for task in task_list:
	if not prev_task is None:
		if not os.path.isfile(os.path.join(prev_task, 'status.txt')):
			print("\n\n{} did not complete successfully".format(prev_task))
			print("Breaking Operation", flush = True)
			break
	os.chdir(task)
	os.system("rm status.txt 2> /dev/null")
	print("Now Running {}".format(task), flush = True)
	os.system('python3 -Wignore main.py --cuda {} && python3 -Wignore main.py --cuda {} --resume --eval > results.txt'.format(args.cuda, args.cuda))
	prev_task = task
	os.chdir('../')