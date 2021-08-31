# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:56:53 2021

@author: Efe
"""


def getElement(n):
  return wd.find_element_by_xpath('//*[@id="TumVerileriGetir"]/tr[' + str(n) + ']/td[11]').text;

def bringLastNDayDeathCases(n): # Returns a list of last 40 days death cases.
  cases = []
  for i in range(1, n+2):
    cases.append( int(getElement(i)) )
  return cases[::-1] #cases.reverse() does not work. Returns None.

def createDates(n):
  froma = 40 - n
  date = []
  for i in range(froma, 41):
    date.append(i)
  return date

def averageOfList(num):
    sumOfNumbers = 0
    for t in num:
        sumOfNumbers = sumOfNumbers + t

    avg = sumOfNumbers / len(num)
    return avg

def leastSquareMethod(dates, deaths):
  meanX = averageOfList(dates)
  meanY = averageOfList(deaths)  

  sumOfXminusMeanXSquare = 0
  sumOfDiff = 0

  for i in range( len(dates) ):
    sumOfXminusMeanXSquare += (dates[i] - meanX)**2
  for i in range( len(deaths) ):
    sumOfDiff += (dates[i] - meanX) * (deaths[i] - meanY)

  b1 = sumOfDiff / sumOfXminusMeanXSquare;
  b0 = (b1 * meanX - meanY) * -1

  return b0, b1

def calculateUsingRegression(b0, b1, val):
  return b0 + b1 * val

def computeCost(X, y, theta):
    m = np.size(X, 0)
    cost = 1/(2*m) * np.sum( (X.dot(theta) - y)**2 )
    return cost

def gradientDescent(X, y, theta, alpha, num_iters):
    # Will return [theta, J_history]
    m = np.size(X, 0)
    n = np.size(theta)
    J_history = []
    for i in range(num_iters):
        #tempT1 = []
        #for j in range(n):
        #    tempT1.append( theta[j] - alpha/m * (np.sum( (X.dot(theta) - y) *  np.reshape(X[:, j], (m,1)) )) )
        #for j in range(n):
        #    theta[j] = tempT1[j]
        # Requires higher learning rate does same thing with above as result
        # Above does not work with scipy.optimize while below works.
        theta = theta - alpha/m * ((1/m) * (X.T.dot(X.dot(theta) - y)))
        J_history.append( computeCost(X, y, theta) )
    return theta, J_history

def mlModel(val, theta):
  return ( calculateUsingRegression(b0, b1, val) * theta[0] + calculateUsingRegression(b2, b3, val ) * theta[1] + calculateUsingRegression(b4, b5, val ) * theta[2] + calculateUsingRegression(b6, b7, val ) * theta[3] )


from selenium import webdriver
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import mysql.connector
import datetime

# Reading the properties.txt file. 
props = {}
count = 0
f_in = open("properties.txt", "r")
lines = list(line for line in (l.strip() for l in f_in) if line)
for line in lines:
    if line.startswith("--"): continue
    elif line.startswith(" ") and not line.strip(): continue
    else:
        count = count + 1
        line = line.replace(" ", "")
        tokens = line.split("=")
        props[tokens[0]] = tokens[1]

from selenium.webdriver.chrome.options import Options
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
figure(figsize=(12, 6), dpi=80)

wd = webdriver.Chrome(executable_path=r""+props["path"],options=options)
wd.get("https://covid19.saglik.gov.tr/TR-66935/genel-koronavirus-tablosu.html")


last5DayCases =  bringLastNDayDeathCases(5)
last10DayCases = bringLastNDayDeathCases(10)
last20DayCases = bringLastNDayDeathCases(20)
last40DayCases = bringLastNDayDeathCases(40)

date1 = createDates(5)
date2 = createDates(10)
date3 = createDates(20)
date4 = createDates(40)


b0, b1 = leastSquareMethod(date4, last40DayCases)
b2, b3 = leastSquareMethod(date3, last20DayCases)
b4, b5 = leastSquareMethod(date2, last10DayCases)
b6, b7 = leastSquareMethod(date1, last5DayCases)

x_axis1 = np.linspace(0, 40, num=39)
y_axis1 = calculateUsingRegression(b0, b1, x_axis1)

x_axis2 = np.linspace(20, 40, num=19)
y_axis2 = calculateUsingRegression(b2, b3, x_axis2 )

x_axis3 = np.linspace(30, 40, num=9)
y_axis3 = calculateUsingRegression(b4, b5, x_axis3)

x_axis4 = np.linspace(35, 40, num=4)
y_axis4 = calculateUsingRegression(b6, b7, x_axis4)

plt.plot(x_axis1, y_axis1, "b")
plt.plot(x_axis2, y_axis2, "y")
plt.plot(x_axis3, y_axis3, "g")
plt.plot(x_axis4, y_axis4, "k")

plt.plot(date4, last40DayCases, "rx", alpha=1, markersize=11)


alpha = 0.00005 # Learning Rate
iterations = 2000
initial_theta = np.array([[0],[0],[0],[0]], dtype=float)

# Machine-1 : Minimizes the cost due to last 40 days.
X = np.zeros((41, 4), dtype=float)
for i in range(41):
 X[i, 0] = calculateUsingRegression(b0, b1, i)
 X[i, 1] = calculateUsingRegression(b2, b3, i)
 X[i, 2] = calculateUsingRegression(b4, b5, i)
 X[i, 3] = calculateUsingRegression(b6, b7, i)

y = np.reshape( np.array(last40DayCases, dtype=float), (41, 1) )
theta, Jhistory = gradientDescent(X, y, initial_theta, alpha, iterations)


# Machine-2 : Minimizes the cost due to last 20 days.
X2 = np.zeros((21, 4), dtype=float)
for i in range(21):
 X2[i, 0] = calculateUsingRegression(b0, b1, 20+i)
 X2[i, 1] = calculateUsingRegression(b2, b3, 20+i)
 X2[i, 2] = calculateUsingRegression(b4, b5, 20+i)
 X2[i, 3] = calculateUsingRegression(b6, b7, 20+i)

initial_theta = np.array([[0],[0],[0],[0]], dtype=float) # Resetting initial theta
y2 = np.reshape( np.array(last20DayCases, dtype=float), (21, 1) )
theta2, Jhistory2 = gradientDescent(X2, y2, initial_theta, alpha, iterations)

# Machine-3 : Minimizes the cost due to last 10 days.
X3 = np.zeros((11, 4), dtype=float)
for i in range(11):
 X3[i, 0] = calculateUsingRegression(b0, b1, 30+i)
 X3[i, 1] = calculateUsingRegression(b2, b3, 30+i)
 X3[i, 2] = calculateUsingRegression(b4, b5, 30+i)
 X3[i, 3] = calculateUsingRegression(b6, b7, 30+i)

initial_theta = np.array([[0],[0],[0],[0]], dtype=float) # Resetting initial theta
y3 = np.reshape( np.array(last10DayCases, dtype=float), (11, 1) )
theta3, Jhistory3 = gradientDescent(X3, y3, initial_theta, alpha, iterations)

print("Machine - 1 : --> Minimizes the cost of last 40 days.")
print("Effects of theta: ")
print("Theta(1) (40 day): " , theta[0], end=" ")
print("Theta(2) (20 day): " , theta[1], end=" ")
print("Theta(3) (10 day): " , theta[2], end=" ")
print("Theta(4) (05 day): " , theta[3])
print("Tolerance :", Jhistory[-1])
print("Machine guess for next announcement : ", mlModel(41, theta)[0] )
print()

print("Machine - 2 : --> Minimizes the cost of last 20 days.")
print("Effects of theta: ")
print("Theta(1) (40 day): " , theta2[0], end=" ")
print("Theta(2) (20 day): " , theta2[1], end=" ")
print("Theta(3) (10 day): " , theta2[2], end=" ")
print("Theta(4) (05 day): " , theta2[3])
print("Tolerance :", Jhistory2[-1] )
print("Machine guess for next announcement : ", mlModel(41, theta2)[0] )
print()

print("Machine - 3 : --> Minimizes the cost of last 10 days.")
print("Effects of theta: ")
print("Theta(1) (40 day): " , theta3[0], end=" ")
print("Theta(2) (20 day): " , theta3[1], end=" ")
print("Theta(3) (10 day): " , theta3[2], end=" ")
print("Theta(4) (05 day): " , theta3[3])
print("Tolerance :", Jhistory3[-1] )
print("Machine guess for next announcement : ", mlModel(41, theta3)[0] )
print()

predictionTheta = None
cur = None
props["machine"] = int(props["machine"])
if props["machine"] == 0:
  if Jhistory[-1] < Jhistory2[-1] and jHistory[-1] < Jhistory3[-1]:
    predictionTheta = theta
  elif Jhistory2[-1] < Jhistory[-1] and Jhistory2[-1] < Jhistory3[-1]:
    predictionTheta = theta2
  else:
    predictionTheta = theta3
elif props["machine"] == 1:
  predictionTheta = theta
elif props["machine"] == 2:
  predictionTheta = theta2
elif props["machine"] == 3:
  predictionTheta = theta3
else:
  raise ValueError('machine value is not set correctly in properties.txt.')
  
print("Prediction: ", mlModel(41, predictionTheta)[0])
#if mlModel(41, theta3)[0] > mlModel(41, theta)[0]:
#  print("Range", mlModel(41, theta)[0], mlModel(41, theta3)[0])
#else:
#  print("Range", mlModel(41, theta3)[0], mlModel(41, theta)[0])

if props["dbOperations"] == "true":
  try:
    mydb = mysql.connector.connect(
    host=""+props["host"],
    user=""+props["user"],
    password=""+props["password"],
    db=""+props["db"]
    )
    cur = mydb.cursor()
    todayDate = datetime.datetime.now()
    todayDate = str(todayDate)[0:10]
    cur.execute("SELECT * from cases ORDER BY date DESC LIMIT 1")
    data = cur.fetchall()
    lastDate = str(data[0][0])
    ratio = last40DayCases[-1]/data[0][1]
    if ratio > 1:
      ratio = 1/ratio
    if not lastDate == todayDate and not data[0][1] == mlModel(41, predictionTheta)[0]:
      cur.execute("UPDATE cases SET realValue = " + str(last40DayCases[-1]) +" , succesRatio = " + str(ratio) + " WHERE date = " + ' "'+ lastDate + '"')
      cur.execute("INSERT INTO `cases` (`date`, `predictedValue`, `realValue`, `succesRatio`) VALUES (NOW(), " + str(mlModel(41, predictionTheta)[0]) + ", DEFAULT, DEFAULT)")
    else:
      pass #print("Nothing is done due to not new data has arrived!")
    mydb.commit()
    mydb.close()
  except:
    print("Check your database!")



wd.close() #Endpoint of program.
wd.quit()

if(props["plot"] == "true"):
  plt.show()
