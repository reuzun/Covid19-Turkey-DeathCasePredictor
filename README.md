# Covid19-Turkey-DeathCasePredictor
This is a project that does below list :

**1.** Creates 4 linear regression line using least squares method according to last 5, 10, 20, 40 day's death cases in Turkey.

**2.** Creates a function using these 4 regression lines. (E.g F(x) = line1(x) * 0.2 + line2(x) * 0.2 + line3(x) * 0.6 + line4(x) * -0.1) No need for sum of coefficients to equal 1.

**3.** Minimizes cost function of that function due to last 40, 20, 10 day. 

**4.** Predicts next days values as minimized function which is minimized for last 20 days according to cost. (You can change it from properties.txt)

**5.** Also saves the predict to database and compares it with the anounced value also saves the succes ratio. (You should enable it from properties.txt)

# Example Result and Ploting
![image](https://user-images.githubusercontent.com/73116832/130816050-d94a6a21-5d0c-46e4-8657-52840dd41438.png)
![image](https://user-images.githubusercontent.com/73116832/130816968-82e07ade-28cf-4247-b1e3-76cfa4316550.png)

# How to run
You can directly run it. If you want to save your predictions and real values to a MySQL database create a database set its name in properties ..etc. Create a table like below:

![image](https://user-images.githubusercontent.com/73116832/130815796-66de8ecc-2cb3-4be5-97a1-b74a176fc135.png)


**Note** : To get rid of Chrome Driver's cmd you should go to your selenium library location and in service.py you should append something in the start method.
```
self.process = subprocess.Popen(cmd, env=self.env,
                                         close_fds=platform.system() != 'Windows',
                                         stdout=self.log_file,
                                         stderr=self.log_file,
                                         stdin=PIPE,
                                         creationflags=134217728)# This is what we append. Just this line
```

# Used libraries
- Selenium
- numPy
- matplotlib
- MySQL

# License
Licensed under [GPL-3.0 License](LICENSE).
