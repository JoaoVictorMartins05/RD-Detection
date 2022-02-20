import os

files = [ f for f in os.listdir("./database/imgSaudavel") ]

DRYRUN=True


for (index,filename) in enumerate(files):
    extension = os.path.splitext(filename)[1]
    newname = "img_%d%s" % (index,extension)
    print ("Renaming %s to %s" % (filename,newname))
    os.rename("./database/imgSaudavel/" +filename,"./database/imgSaudavel/" + newname)

    

files = [ f for f in os.listdir("./database/imgMild") ]

DRYRUN=True

for (index,filename) in enumerate(files):
    extension = os.path.splitext(filename)[1] 
    newname = "img_%d%s" % (index,extension)
    print ("Renaming %s to %s" % (filename,newname))
    os.rename("./database/imgMild/" +filename,"./database/imgMild/" + newname)  

files = [ f for f in os.listdir("./database/imgModerate/") ]

DRYRUN=True

for (index,filename) in enumerate(files):
    extension = os.path.splitext(filename)[1] 
    newname = "img_%d%s" % (index,extension)
    print ("Renaming %s to %s" % (filename,newname))
    os.rename("./database/imgModerate/" +filename,"./database/imgModerate/" + newname)


files = [ f for f in os.listdir("./database/imgProliferate") ]

DRYRUN=True

for (index,filename) in enumerate(files):
    extension = os.path.splitext(filename)[1] 
    newname = "img_%d%s" % (index,extension)
    print ("Renaming %s to %s" % (filename,newname))
    os.rename("./database/imgProliferate/" +filename,"./database/imgProliferate/" + newname)    
    

files = [ f for f in os.listdir("./database/imgSevere") ]

DRYRUN=True

for (index,filename) in enumerate(files):
    extension = os.path.splitext(filename)[1] 
    newname = "img_%d%s" % (index,extension)
    print ("Renaming %s to %s" % (filename,newname))
    os.rename("./database/imgSevere/" +filename,"./database/imgSevere/" + newname)