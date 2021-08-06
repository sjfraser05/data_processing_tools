import datetime
import math
import numpy as np
import glob, os
import easygui
import csv
import PyPDF2
import re

directory = easygui.diropenbox()
os.chdir(directory)
dirList = glob.glob("*.pdf")
nList = len(dirList)
OSLarray =  np.zeros((1,13))

for count, iFile in enumerate(dirList):
    pdfFileObj = open(iFile, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    #print(pdfReader.numPages)
    pageObj1 = pdfReader.getPage(0)
    pageObj2 = pdfReader.getPage(1)
    # extracting text from page
    pageText1 = pageObj1.extractText().replace("\n", "")
    pageText2 = pageObj2.extractText().replace("\n", "")
    #print(pageText1)
    #print(pageText2)
    # closing the pdf file object
    pdfFileObj.close()
    #Remove spaces from all outputs except names
    #Search for tag before WM, replace all spaces, and remove alpabetical characters
    try:
        HSMstart = re.search("WM", pageText1).end()
        HSMend = re.search("STORE ADDRESS:", pageText1).start()
        StoreNum = pageText1[HSMstart:HSMend].replace(" ", "")
    except:
        StoreNum = "ERROR"

    try:
        Teamstart = re.search("TEAM NAME: ", pageText1).end()
        Teamend = re.search("MANAGER NAME", pageText1).start()
        TeamName = pageText1[Teamstart:Teamend].replace(" ", "")
    except:
        TeamName = "ERROR"

    try:
        WARPstart = re.search("EMPLOYEE ID: ", pageText1).end()
        WARPend = re.search("ASSOCIATE ID", pageText1).start()
        EmpID = pageText1[WARPstart:WARPend].replace(" ", "")
        WARPID = "OSL" + pageText1[WARPstart:WARPend].replace(" ", "")

    except:
        WARPID = "ERROR"

    try:
        LastNamestart = re.search("Last Name: ", pageText2).end()
        LastNameend = re.search("Birthdate:", pageText2).start()
        LastName = pageText2[LastNamestart:LastNameend].strip()

    except:
        LastName = "ERROR"

    try:
        FirstNamestart = re.search("First Name: ", pageText2).end()
        FirstNameend = re.search("\  Last Name:", pageText2).start()
        FirstName = pageText2[FirstNamestart:FirstNameend-1].strip()
        res = re.search(r'\W+', FirstName).start()
        FirstName = FirstName[0: res]

    except:
        FirstName = "ERROR"

    try:
        Emailstart = re.search("EMAIL: ", pageText1).end()
        Emailend = re.search("EMPLOYEE PHONE #:", pageText1).start()
        Email = pageText1[Emailstart:Emailend].replace(" ", "")
    except:
        Email= "ERROR"

    try:
        Phonestart = re.search("EMPLOYEE PHONE #: ", pageText1).end()
        Phoneend = re.search("PROGRAM:", pageText1).start()
        Phonenum = pageText1[Phonestart:Phoneend].replace("-", "").replace("(", "").replace(")", "").replace(".", "").replace(" ", "")

    except:
        Phonenum= "ERROR"

    try:
        Titlestart = re.search("JOB TITLE: ", pageText1).end()
        Titleend = re.search("POSITION ID:", pageText1).start()
        Title = pageText1[Titlestart:Titleend].strip()
    except:
        Title= "ERROR"

    try:
        AssIDstart = re.search("ASSOCIATE ID: ", pageText1).end()
        AssIDend = re.search("TEAM NAME:", pageText1).start()
        AssID = pageText1[AssIDstart:AssIDend].replace(" ", "")
    except:
        AssID= "ERROR"

    try:
        POSIDstart = re.search("POSITION ID: ", pageText1).end()
        POSIDend = re.search("EMPLOYEE ID:", pageText1).start()
        POSID = pageText1[POSIDstart:POSIDend].replace(" ", "")
    except:
        POSID = "ERROR"

    try:
        StDateStart = re.search("START DATE:", pageText1).end()
        StDateEnd = re.search("EMAIL:", pageText1).start()
        StartDate = pageText1[StDateStart:StDateEnd].replace(" ", "")
    except:
        StartDate = "ERROR"

    excelList = np.asarray([TeamName, WARPID, LastName, FirstName, Email, Phonenum, StoreNum, Title, AssID, POSID, ' ', EmpID, StartDate])
    excelList = np.resize(excelList, (1, len(excelList)))
    #print(excelList)
    OSLarray = np.vstack((OSLarray, excelList))
    #print(OSLarray)

OSLarray = np.delete(OSLarray, (0), axis=0)
today = datetime.date.today()
fileName = "OSL input Array " + str(today) + ".csv"
np.savetxt(fileName, OSLarray, delimiter=',', fmt='% s')
