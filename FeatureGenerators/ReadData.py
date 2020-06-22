import numpy as np
import csv, operator, sys, os

'''def readResultsFiles(dataSet, scoring_function):
    
    resultsPath = "/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/forests/rootedFrequentTrees/"
    
    res_list_dt=[]
    res_list_rf=[]
    res_list_normal=[]
    res_list_rf_5=[]
    with open(resultsPath+dataSet+'/RF_'+str(5)+'_'+scoring_function+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        line_count = 0
        for row in csv_reader:
            rowStr = str(row).split(',')
            if line_count < 25:
                res_list_rf_5.append(rowStr[1])
                res_list_normal.append(rowStr[2])
                res_list_dt.append(rowStr[3])
                res_list_rf.append(rowStr[4])
                line_count+=1
    csv_file.close()
    
    res_list_rf_10=[]
    with open(resultsPath+dataSet+'/RF_'+str(10)+'_'+scoring_function+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        line_count = 0
        for row in csv_reader:
            rowStr = str(row).split(',')
            if line_count < 25:
                res_list_rf_10.append(rowStr[1])
                line_count+=1
    csv_file.close()
    
    res_list_rf_15=[]
    with open(resultsPath+dataSet+'/RF_'+str(15)+'_'+scoring_function+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        line_count = 0
        for row in csv_reader:
            rowStr = str(row).split(',')
            if line_count < 25:
                res_list_rf_15.append(rowStr[1])
                line_count+=1
    csv_file.close()
    
    res_list_rf_20=[]
    with open(resultsPath+dataSet+'/RF_'+str(20)+'_'+scoring_function+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        line_count = 0
        for row in csv_reader:
            rowStr = str(row).split(',')
            if line_count < 25:
                res_list_rf_20.append(rowStr[1])
                line_count+=1
    csv_file.close()
    
    f= open(resultsPath+dataSet+'/RF_all'+'_'+scoring_function+'.csv',"a")
    for i in range(25):
            f.write(res_list_rf_5[i]+','+res_list_rf_10[i]+','+res_list_rf_15[i]+','+res_list_rf_20[i]+','
                    
'''                                   
   
    
    
    


def readData(dataSetStr):
	path = "/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/"
	X = []
	Y = []
	if (dataSetStr =='adult'):
		f = open(path + "adult/adult.data")
		for row in f:
			if (len(row) > 1):
				entries = row.replace("\n", "").replace(" ", "").split(",")

				x = getFeatureVectorAdult(entries)

				if (entries[-1] == "<=50K"):
					y = 0
				else:
					y = 1

				Y.append(y)
				X.append(x)
		X = np.array(X).astype(dtype=np.int32)
		Y = np.array(Y)
		f.close()
	elif(dataSetStr == 'wine-quality'):
		red = np.genfromtxt(
			"/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/wine-quality/winequality-red.csv",
			delimiter=';', skip_header=1)
		white = np.genfromtxt(
			"/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/wine-quality/winequality-white.csv",
			delimiter=';', skip_header=1)
		X = np.vstack((red[:, :-1], white[:, :-1])).astype(dtype=np.float32)
		Y = np.concatenate((red[:, -1], white[:, -1]))
		Y = Y - min(Y)


	return (X,Y)




def readDataAdult(type):
	path = "/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/"
	X = []
	Y = []       

	f = open(path + "adult/adult."+type)
	for row in f:
		if (len(row) > 1):
			entries = row.replace("\n", "").replace(" ", "").split(",")

			x = getFeatureVectorAdult(entries)

			if (entries[-1] == "<=50K"):
				y = 0
			else:
				y = 1

			Y.append(y)
			X.append(x)
	X = np.array(X).astype(dtype=np.int32)
	Y = np.array(Y)
	f.close()
	return (X, Y)
	

def readWine():
	X = []
	Y = []
	red = np.genfromtxt(
		"/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/wine-quality/winequality-red.csv",
		delimiter=';', skip_header=1)
	white = np.genfromtxt("/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/wine-quality/winequality-white.csv", delimiter=';', skip_header=1)
	X = np.vstack((red[:, :-1], white[:, :-1])).astype(dtype=np.float32)
	Y = np.concatenate((red[:, -1], white[:, -1]))
	Y = Y - min(Y)
	return (X, Y)

def readWineTest():
	X = []
	Y = []
	red = np.genfromtxt(
		"/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/wine-quality/test.csv",
		delimiter=';', skip_header=1)
	white = np.genfromtxt("/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/wine-quality/test2.csv", delimiter=';', skip_header=1)
	X = np.vstack((red[:, :-1], white[:, :-1])).astype(dtype=np.float32)
	Y = np.concatenate((red[:, -1], white[:, -1]))
	Y = Y - min(Y)
	return (X, Y)

def getFeatureVectorAdult(entries):
	x = []
	x.append(float(entries[0])) # age = continous

	workclass = [0 for i in range(8)]
	if entries[1] == "Private":
		workclass[0] = 1
	elif entries[1] == "Self-emp-not-inc":
		workclass[1] = 1
	elif entries[1] == "Self-emp-inc":
		workclass[2] = 1
	elif entries[1] == "Federal-gov":
		workclass[3] = 1
	elif entries[1] == "Local-gov":
		workclass[4] = 1
	elif entries[1] == "State-gov":
		workclass[5] = 1
	elif entries[1] == "Without-pay":
		workclass[6] = 1
	else: #Never-worked
		workclass[7] = 1
	x.extend(workclass)
	x.append(float(entries[2]))

	education = [0 for i in range(16)]
	if entries[3] == "Bachelors":
		education[0] = 1
	elif entries[3] == "Some-college":
		education[1] = 1
	elif entries[3] == "11th":
		education[2] = 1
	elif entries[3] == "HS-grad":
		education[3] = 1
	elif entries[3] == "Prof-school":
		education[4] = 1
	elif entries[3] == "Assoc-acdm":
		education[5] = 1
	elif entries[3] == "Assoc-voc":
		education[6] = 1
	elif entries[3] == "9th":
		education[7] = 1
	elif entries[3] == "7th-8th":
		education[8] = 1
	elif entries[3] == "12th":
		education[9] = 1
	elif entries[3] == "Masters":
		education[10] = 1
	elif entries[3] == "1st-4th":
		education[11] = 1
	elif entries[3] == "10th":
		education[12] = 1
	elif entries[3] == "Doctorate":
		education[13] = 1
	elif entries[3] == "5th-6th":
		education[14] = 1
	if entries[3] == "Preschool":
		education[15] = 1

	x.extend(education)
	x.append(float(entries[4]))

	marital = [0 for i in range(7)]
	if entries[5] == "Married-civ-spouse":
		marital[0] = 1
	elif entries[5] == "Divorced":
		marital[1] = 1
	elif entries[5] == "Never-married":
		marital[2] = 1
	elif entries[5] == "Separated":
		marital[3] = 1
	elif entries[5] == "Widowed":
		marital[4] = 1
	elif entries[5] == "Married-spouse-absent":
		marital[5] = 1
	else:
		marital[6] = 1
	x.extend(marital)

	occupation = [0 for i in range(14)]
	if entries[6] == "Tech-support":
		occupation[0] = 1
	elif entries[6] == "Craft-repair":
		occupation[1] = 1
	elif entries[6] == "Other-service":
		occupation[2] = 1
	elif entries[6] == "Sales":
		occupation[3] = 1
	elif entries[6] == "Exec-managerial":
		occupation[4] = 1
	elif entries[6] == "Prof-specialty":
		occupation[5] = 1
	elif entries[6] == "Handlers-cleaners":
		occupation[6] = 1
	elif entries[6] == "Machine-op-inspct":
		occupation[7] = 1
	elif entries[6] == "Adm-clerical":
		occupation[8] = 1
	elif entries[6] == "Farming-fishing":
		occupation[9] = 1
	elif entries[6] == "Transport-moving":
		occupation[10] = 1
	elif entries[6] == "Priv-house-serv":
		occupation[11] = 1
	elif entries[6] == "Protective-serv":
		occupation[12] = 1
	else:
		occupation[13] = 1
	x.extend(occupation)

	relationship = [0 for i in range(6)]
	if entries[7] == "Wife":
		relationship[0] = 1
	elif entries[7] == "Own-child":
		relationship[1] = 1
	elif entries[7] == "Husband":
		relationship[2] = 1
	elif entries[7] == "Not-in-family":
		relationship[3] = 1
	elif entries[7] == "Other-relative":
		relationship[4] = 1
	else:
		relationship[5] = 1
	x.extend(relationship)

	race = [0 for i in range(5)]
	if entries[8] == "White":
		race[0] = 1
	elif entries[8] == "Asian-Pac-Islander":
		race[1] = 1
	elif entries[8] == "Amer-Indian-Eskimo":
		race[2] = 1
	elif entries[8] == "Other":
		race[3] = 1
	else:
		race[4] = 1
	x.extend(race)

	if (entries[9] == "Male"):
		x.extend([1,0])
	else:
		x.extend([0,1])

	x.append(float(entries[10]))
	x.append(float(entries[11]))
	x.append(float(entries[12]))

	native = [0 for i in range(42)]
	if entries[14] == "United-States":
		native[1] = 1
	elif entries[14] == "Cambodia":
		native[2] = 1
	elif entries[14] == "England":
		native[3] = 1
	elif entries[14] == "Puerto-Rico":
		native[4] = 1
	elif entries[14] == "Canada":
		native[5] = 1
	elif entries[14] == "Germany":
		native[6] = 1
	elif entries[14] == "Outlying-US(Guam-USVI-etc)":
		native[7] = 1
	elif entries[14] == "India":
		native[8] = 1
	elif entries[14] == "Japan":
		native[9] = 1
	elif entries[14] == "Greece":
		native[10] = 1
	elif entries[14] == "South":
		native[11] = 1
	elif entries[14] == "China":
		native[12] = 1
	elif entries[14] == "Cuba":
		native[13] = 1
	elif entries[14] == "Iran":
		native[14] = 1
	elif entries[14] == "Honduras":
		native[15] = 1
	elif entries[14] == "Philippines":
		native[16] = 1
	elif entries[14] == "Italy":
		native[17] = 1
	elif entries[14] == "Poland":
		native[18] = 1
	elif entries[14] == "Jamaica":
		native[19] = 1
	elif entries[14] == "Vietnam":
		native[20] = 1
	elif entries[14] == "Mexico":
		native[21] = 1
	elif entries[14] == "Portugal":
		native[22] = 1
	elif entries[14] == "Ireland":
		native[23] = 1
	elif entries[14] == "France":
		native[24] = 1
	elif entries[14] == "Dominican-Republic":
		native[25] = 1
	elif entries[14] == "Laos":
		native[26] = 1
	elif entries[14] == "Ecuador":
		native[27] = 1
	elif entries[14] == "Taiwan":
		native[28] = 1
	elif entries[14] == "Haiti":
		native[29] = 1
	elif entries[14] == "Columbia":
		native[30] = 1
	elif entries[14] == "Hungary":
		native[31] = 1
	elif entries[14] == "Guatemala":
		native[32] = 1
	elif entries[14] == "Nicaragua":
		native[33] = 1
	elif entries[14] == "Scotland":
		native[34] = 1
	elif entries[14] == "Thailand":
		native[35] = 1
	elif entries[14] == "Yugoslavia":
		native[36] = 1
	elif entries[14] == "El-Salvador":
		native[37] = 1
	elif entries[14] == "Trinadad&Tobago":
		native[38] = 1
	elif entries[14] == "Peru":
		native[39] = 1
	elif entries[14] == "Hong":
		native[40] = 1
	else:
		native[41] = 1

	return x



def getFeatureVectorBank(entries):
	x = []
	x.append(float(entries[0])) # age = continous

	job = [0 for i in range(12)]
	if entries[1] == "admin.":
		job[0] = 1
	elif entries[1] == "blue-collar":
		job[1] = 1
	elif entries[1] == "entrepreneur":
		job[2] = 1
	elif entries[1] == "housemaid":
		job[3] = 1
	elif entries[1] == "management":
		job[4] = 1
	elif entries[1] == "retired":
		job[5] = 1
	elif entries[1] == "self-employed":
		job[6] = 1
	elif entries[1] == "services":
		job[7] = 1
	elif entries[1] == "student":
		job[8] = 1
	elif entries[1] == "technician":
		job[9] = 1
	elif entries[1] == "unemployed":
		job[10] = 1
	else:
		job[11] = 1

	x.extend(job)

	martial = [0 for i in range(4)]
	if entries[2] == "divorced":
		martial[0] = 1
	elif entries[2] == "married":
		martial[1] = 1
	elif entries[2] == "single":
		martial[2] = 1
	else:
		martial[3] = 1

	education = [0 for i in range(8)]
	if entries[3] == "basic.4y":
		education[0] = 1
	elif entries[3] == "basic.6y":
		education[1] = 1
	elif entries[3] == "basic.9y":
		education[2] = 1
	elif entries[3] == "high.school":
		education[3] = 1
	elif entries[3] == "illiterate":
		education[4] = 1
	elif entries[3] == "professional.course":
		education[5] = 1
	elif entries[3] == "university.degree":
		education[6] = 1
	else:
		education[7] = 1
	x.extend(education)


	if entries[4] == "no":
		x.extend([1,0,0])
	elif entries[4] == "yes":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	if entries[5] == "no":
		x.extend([1,0,0])
	elif entries[5] == "yes":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	if entries[6] == "no":
		x.extend([1,0,0])
	elif entries[6] == "yes":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	if entries[7] == "telephone":
		x.append(0)
	else:
		x.append(1)

	month = [0 for i in range(12)]
	if entries[8] == "jan":
		month[0] = 1
	elif entries[8] == "feb":
		month[1] = 1
	elif entries[8] == "mar":
		month[2] = 1
	elif entries[8] == "apr":
		month[3] = 1
	elif entries[8] == "may":
		month[4] = 1
	elif entries[8] == "jun":
		month[5] = 1
	elif entries[8] == "jul":
		month[6] = 1
	elif entries[8] == "aug":
		month[7] = 1
	elif entries[8] == "sep":
		month[8] = 1
	elif entries[8] == "oct":
		month[9] = 1
	elif entries[8] == "nov":
		month[10] = 1
	else:
		month[11] = 1
	x.extend(month)

	day = [0 for i in range(5)]
	if entries[9] == "mon":
		day[0] = 1
	elif entries[9] == "tue":
		day[1] = 1
	elif entries[9] == "wed":
		day[2] = 1
	elif entries[9] == "thu":
		day[3] = 1
	else:
		day[4] = 1
	x.extend(day)

	#x.append(float(entries[10]))
	x.append(int(float(entries[11])*1000))
	x.append(int(float(entries[12])*1000))
	x.append(int(float(entries[13])*1000))

	if entries[14] == "failure":
		x.extend([1,0,0])
	elif entries[14] == "nonexistent":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	x.append(int(float(entries[15])*1000))
	x.append(int(float(entries[16])*1000))
	x.append(int(float(entries[17])*1000))
	x.append(int(float(entries[18])*1000))
	x.append(int(float(entries[19])*1000))

	return x