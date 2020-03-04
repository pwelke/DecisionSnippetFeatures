import json

dataset = "wine-quality"
variant = "WithLeafEdges"
pattern_max_size = 6
jsonPath = "/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data"
patternsPath = "/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/InferenceComparison"

a_dict = {'a': 'a value', 'b': {'b.1': 'b.1 value'}}
 
def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

#results = extract(obj, arr, key)

def traverse(data):
    print (' ' * str(traverse.level) + data['leftChild'])
    for kid in data['kids']:
        traverse.level += 1
        traverse(kid)
        traverse.level -= 1
        

def iterate(dictionary, patternText):
    #print(patternText)
    
    if 'prediction' in dictionary:
        patternText+= ' leaf '
        #return ' leaf '
    
    else:
        if 'feature' in dictionary:
            patternText += str(dictionary['feature'])+'<'+str(dictionary['split'])
        
        if 'leftChild' in dictionary:
        
        
            #print(dictionary['leftChild'])
            patternText+= ' ( leftChild '
        
            iterate(dictionary['leftChild'],patternText)
            patternText+= ' ) '
            
        if 'rightChild' in dictionary:
            patternText+= ' ( rightChild '
            iterate(dictionary['rightChild'],patternText)
            patternText+= ' ) '
        
        
        
        
    
    
        
        #print(dictionary['feature'])
        
    
         
         
        
    
        
    return patternText
   
        

patternList=[]

def iterdict(d,patternText):
    
   patternText = ''
   print(patternText)
   for k,v in d.items():
    if (k == 'feature'):
        patternText += str(v)+'<'
        
    if (k == 'split'):
        patternText += str(v)
        
    if (k == 'leftChild'):
        patternText += ' ( leftChild '
        patternText+= iterdict(d['leftChild'],patternText)
        patternText+= ' ) '
        
    if (k == 'rightChild'):
        patternText += ' ( rightChild '
        patternText+= iterdict(d['rightChild'],patternText)
        patternText+= ' ) '
        
    if (k == 'prediction'):
        patternText += ' leaf '    
        
        
   return patternText
        
            
   
           
    
            
            
with open('DT_5_test.json') as json_file:
         data = json.load(json_file)
         
         #list = [i for i in range(len(line)) if line.startswith('"id"', i)]
         for tree in range(0,len(data)):
                
             patternText = ''
             patternText = iterdict(data[0],patternText)
             patternText='1\t'+str(tree+1)+'\t'+patternText
             #patternText = iterate(data[0],patternText)
             #traverse.level = 1
             #traverse(data[0])
             
             #print(patternText)
             #print(data[tree]['id'])
             #print(data[tree]['feature'])
             #print(data[tree]['split'])
             #print(data[tree])
             #patternText += '1\t'+str(tree)+'\t'+str(data[tree]['feature'])+'<'+str(data[tree]['split'])
             #if 'leftChild' in data[tree]:
             #  patternText+= ' ( leftChild'
                
             patternList.append(patternText)
             #print(data[tree]['leftChild']['id'])
            
            
            
         
        
         #nodes_count_list.append('RF_'+str(rf_depth)+'_t'+str(frequency)+','+str(len(list))+',\n')
                
        
            #rowStr = str(row).split(',')
            
json_file.close()
for pattern in patternList:
    print(pattern)
    
f= open('DT_5.patterns',"w")
for pattern in patternList:
        f.write(pattern)
f.close()    



