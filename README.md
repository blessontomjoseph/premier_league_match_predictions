# Premier League Predictions :england:
![poster](rd_fils/RemoteQueasyAmericancrow-size_restricted.gif)
### Data
Features:
Premier league matche details from 2000 to 2017 season. Featurs are different perfomance metrices and general match evaluation attributes.
<!-- ![poster](rd_fils/features.jpg) -->

| Feature | Feature Abbreviation     | 
| :-------- | :------- | 
|Season  | year of season  | 
| Datetime |datetime details  | 
| HomeTeam |  home team name| 
|  AwayTeam| away team name | 
| FTHG | full time home goals | 
| FTAG |  full time away goals| 
| HTHG |  half time home goals| 
|  HTAG| half time away goals | 
| FTR | full time results | 
|  HTR|  half time results| 
|  Referee|  name of referee | 
| HS | home team shots | 
| AS |  away team shots| 
| HST |  home team shots on target| 
|  AST|  away team shots on target| 
|  HC|  home team corners| 
|  AC| away team corners | 
|  HF| home team fouls | 
|  AF|  away team fouls| 
| HY |  home team yellow cards| 
| AY | away team yellow cards| 
| HR |  home team red cards| 
| AR | away team red cards| 


### Design
![poster](rd_fils/workflow.jpg)

### Feature Engineering
many of the features are potential data leakage factors, since most of them are recorder in the match. so, the approach is to change the features into  rolling features. so every time we look at a match we would see their current profile/form delivered through the attibutes
![poster](rd_fils/features.jpg)



#### Attack 
all of the home and away features 

#### Defence 

### Algorithm

