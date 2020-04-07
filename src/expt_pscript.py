#!/usr/bin/env python3
import subprocess	

# log = open('outfiles/rq1.1/rq1.1_robust_ls.out','a+')
# log.write("Robust optimizer with learned support (Random Forest Regressor)\n")
# log.flush()
# for j in ['4','7','10','15']:
# 	for k in ['1','2','3','4','5']:
# 		print("PROCESSING DISEASES: ", j, " FILE NUMBER: ",k)
# 		log.write("########################################################################\n")
# 		log.write("PROCESSING DISEASES: "+str(j)+" FILE NUMBER: "+str(k)+"\n")
# 		log.write("########################################################################\n")
# 		log.flush()
# 		subprocess.call(['python', 'rq1.1.py', j, k], stdout=log, shell=True)
# log.close()


# for i in ['4','7','10','15']:
# 	for j in ['1','2','3','4','5']:
# 		print("###################################################")
# 		print("PROCESSING DISEASES: ", i, "FILE NUMBER: ", j)
# 		print("###################################################")
# 		subprocess.call(['python', 'rq1.1.py', i, j], shell=True)

# Quick trial
# for i in range(5,10):
# 	for j in range(5,10):
# 		print("##############################################")
# 		print("PROCESSING DATASET: ", str(i), " FILE NUMBER: ", str(j))
# 		print("##############################################")
# 		cmd = "python rq5.2.py 20 "+str(i)+" "+str(10)+" "+str(0.005)+" "+str(j)
# 		subprocess.call(cmd, shell=True)

# on swarm
# for i in ['10','20','30','40','50','60']:
# 	for j in ['4','7','10','15']:
# 		print("###################################################")
# 		print("PROCESSING DATASET: ", i, "DISEASES: ", j)
# 		print("###################################################")
# 		subprocess.call(['python vis_rq2.2.py'+' '+j+' '+i], shell=True)

# log = open('outfiles/rq2.1/rq2.1_robust_ls.out','a+')
# log.write("Robust optimizer with width = 1 and Learned support\n")
# log.flush()
# for j in ['4','7','10','15']:
# 	for k in ['1','2','3','4','5']:
# 		for p in ['0','1']:
# 			print("PROCESSING DISEASES: ", j, " FILE NUMBER: ",k, " PERT: ", p)
# 			log.write("########################################################################\n")
# 			log.write("PROCESSING DISEASES: "+str(j)+" FILE NUMBER: "+str(k)+" PERT: "+str(p)+"\n")
# 			log.write("########################################################################\n")
# 			log.flush()
# 			subprocess.call(['python', 'rq2.1.py', j, k, p], stdout=log, shell=True)
# log.close()

log = open('outfiles/rq3.2/rq3.2_robust_ls.out','a+')
log.write("Robust optimizer with width = 1 and Learned support compared to Empirical Probabilities\n")
log.flush()
for j in ['4','7','10','15']:
	for k in ['1','2','3','4','5']:
		print("PROCESSING DISEASES: ", j, " FILE NUMBER: ",k)
		log.write("########################################################################\n")
		log.write("PROCESSING DISEASES: "+str(j)+" FILE NUMBER: "+str(k)+"\n")
		log.write("########################################################################\n")
		log.flush()
		subprocess.call(['python', 'rq3.2.py', j, k], stdout=log, shell=True)
log.close()

# log = open('outfiles/rq2.2/rq2.2_widthtest.out','a+')
# log.write("Getting the best width for the robust optimizer\n")
# log.flush()
# for j in ['4','7','10','15']:
# 	for k in ['1','2','3','4','5']:
# 		for p in ['0','1']:
# 			print("PROCESSING DISEASES: ", j, " FILE NUMBER: ",k, " PERT: ", p)
# 			log.write("########################################################################\n")
# 			log.write("PROCESSING DISEASES: "+str(j)+" FILE NUMBER: "+str(k)+" PERT: "+str(p)+"\n")
# 			log.write("########################################################################\n")
# 			log.flush()
# 			subprocess.call(['python', 'rq2.2.py', j, k, p], stdout=log, shell=True)
# log.close()

# Width testing for different datasets 
# log = open('outfiles/rq2.2/rq2.2_widthtest.out','a+')
# log.write("Getting the best width for the robust optimizer\n")
# log.flush()
# for i in ['10','20','30','40','50','60']:
# 	for j in ['4','7','10']:
# 		for k in ['1','2','3','4','5']:
# 			for p in ['0','1']:
# 				print("PROCESSING DATASET: ", i, " DISEASES: ", j, " FILE NUMBER: ",k, " PERT: ", p)
# 				log.write("##############################################################################\n")
# 				log.write("PROCESSING DATASET: "+str(i)+" DISEASES: "+str(j)+" FILE NUMBER: "+str(k)+" PERT: "+str(p)+"\n")
# 				log.write("##############################################################################\n")
# 				log.flush()
# 				subprocess.call(['python', 'rq2.2.py', j, k, p, i], stdout=log, shell=True)
# log.close()

# log = open('outfiles/rq7.2/rq7.2_alldataset.out','a+')
# log.write("Comparing Zero Atom Probabilities in the robust vs. non-robust case\n")
# log.flush()
# for i in ['10','20','30','40','50','60']:
# 	for j in ['4','7','10']:
# 		for k in ['1','2','3','4','5']:
# 			print("###################################################")
# 			print("PROCESSING DATASET: ", i, " DISEASES: ", j, " FILE NUMBER: ",k)
# 			print("###################################################")
# 			log.write("###################################################\n")
# 			log.write("PROCESSING DATASET: "+str(i)+" DISEASES: "+str(j)+" FILE NUMBER: "+str(k)+"\n")
# 			log.write("###################################################\n")
# 			log.flush()
# 			subprocess.call(['python', 'rq7.2.py', j, k, i], stdout=log, shell=True)
# log.close()

# log = open('outfiles/rq1.1/rq1.1_pr5.out','a')
# log.write("Robust optimizer, learned support with polynomial model of degree 5\n")
# log.flush()
# for i in ['4','7','10','15']:
# 	for k in ['1','2','3','4','5']:
# 		print("###################################################")
# 		print("PROCESSING DISEASES: ", i, " FILE NUMBER: ",k)
# 		print("###################################################")
# 		log.write("###################################################\n")
# 		log.write("PROCESSING DISEASES: "+str(i)+" FILE NUMBER: "+str(k)+'\n')
# 		log.write("###################################################\n")
# 		log.flush()
# 		subprocess.call(['python', 'rq1.1.py', i, k], stdout=log, shell=True)
# log.close()

# log = open('outfiles/rq1.3/rq1.1_pr2.out','a')
# log.write("Robust optimizer, learned support with polynomial model of degree 2\n")
# log.flush()
# for d in ['30','31','32','33','34','35']:
# 	for i in ['4','7','10','15']:
# 		for k in ['1','2','3','4','5']:
# 			print("###################################################")
# 			print("PROCESSING DISEASES: ", i, " FILE NUMBER: ",k, " DATASET: ", d)
# 			print("###################################################")
# 			log.write("###################################################\n")
# 			log.write("PROCESSING DISEASES: "+str(i)+" FILE NUMBER: "+str(k)+" DATASET: "+d+'\n')
# 			log.write("###################################################\n")
# 			log.flush()
# 			subprocess.call(['python', 'rq1.3.2.py', i, k, d], stdout=log, shell=True)
# log.close()

# log = open('outfiles/rq1.3/rq1.1_pr5.out','a')
# log.write("Robust optimizer, learned support with polynomial model of degree 5\n")
# log.flush()
# for d in ['30','31','32','33','34','35']:
# 	for i in ['4','7','10','15']:
# 		for k in ['1','2','3','4','5']:
# 			print("###################################################")
# 			print("PROCESSING DISEASES: ", i, " FILE NUMBER: ",k, " DATASET: ", d)
# 			print("###################################################")
# 			log.write("###################################################\n")
# 			log.write("PROCESSING DISEASES: "+str(i)+" FILE NUMBER: "+str(k)+" DATASET: "+d+'\n')
# 			log.write("###################################################\n")
# 			log.flush()
# 			subprocess.call(['python', 'rq1.3.3.py', i, k, d], stdout=log, shell=True)
# log.close()

# log = open('outfiles/rq1.3/rq1.1_rfr.out','a')
# log.write("Robust optimizer, learned support with a random forest regressor\n")
# log.flush()
# for d in ['30','31','32','33','34','35']:
# 	for i in ['4','7','10','15']:
# 		for k in ['1','2','3','4','5']:
# 			print("###################################################")
# 			print("PROCESSING DISEASES: ", i, " FILE NUMBER: ",k, " DATASET: ", d)
# 			print("###################################################")
# 			log.write("###################################################\n")
# 			log.write("PROCESSING DISEASES: "+str(i)+" FILE NUMBER: "+str(k)+" DATASET: "+d+'\n')
# 			log.write("###################################################\n")
# 			log.flush()
# 			subprocess.call(['python', 'rq1.3.4.py', i, k, d], stdout=log, shell=True)
# log.close()

