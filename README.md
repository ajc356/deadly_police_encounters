# Deadly Police Encounters

Amanda Cheney  
Metis Project 3  
October 28, 2020  

**Motivation**

Between January 1, 2013 and October 12, 2020, 8,485 people were killed by police in the United States. 51% of these people have been people of color – people who are Black, Latinx, Native American, Asian, Pacific Islander - this despite the fact that these racial and ethnic groups comprise less than 40% of the overall US population.   

Therefore, in this project, I focus on deadly police encounters where the person whose life was lost was a person of color and I seek to understand where these deadly encounters happen.  

**Guiding Question** 

Where do deadly police encounters occur where the person whose life was lost was a person of color? 

**Objective** 

Build a classification model that can identify where deadly encounters are likely to happen based on socio-economic characteristics of the communities in which they occur. 

**Data Sources** 

The [Mapping Police Violence](https://mappingpoliceviolence.org/) project which has comprehensive data on nearly all deadly police encounters that have occurred in the US since 2013 -- including information about the precise location of the killing as well as information about the victim’s race. 

[American Community Survey Data](https://www.census.gov/programs-surveys/acs) from the US Census Bureau which provides detailed population information on a yearly basis on topics not covered by the decennial census. In my case, I am particularly interested in those relating to socio-economic indicators like education, unemployment, rates of health insurance coverage, use of food stamps, computer ownership and access to the internet – nearly all of which the ACS offers data on by racial/ ethnic categories. 

Since the Mapping Police Violence data has zip code information for its entire dataset of all deadly encounters – my observations are individual zip codes. So for all zip codes in entire the United States, my model seeks to explain which zip codes have deadly encounters in which people of color are killed by police and which zip codes do not by comparing their socio-economic characteristics.  

**Methods**

1. Collected zip code level ACS data using the [census.gov](https://api.census.gov/) API.
2. Engineered features to indicate for all zip codes in the United States whether or not people/ a person of color has been killed by police in that zip code. 
3. Tested 5 different classification models - KNN, Logistic Regression, Random Forest, SVC.
4. Evaluated different sampling techniques to accommodate class imbalance - RandomUnderSampler, RandomOverSampler, SMOTE, ADASYN. ADASYN offered the best performance. 
5. Optimized, evaluated and selected best model based on which model offered the best recall score for the positive/minority class (where people of color are killed by police). I put special emphasis on ensuring that recall for the minority class is as high as possible even if  precision is low because compared to the cost of a human life, which economists estimate is $10 million and that person's family would regard as priceless, the cost of falsely identifying a zip code as having a deadly encounter when it does not is several orders of magnitude less important than the cost of falsely predicting that a zip code does not have a deadly encounter when it in fact has. 
   * The best model ended up being Logistic Regression which correctly classifies 87.5% of deadly encounters in the holdout test set.
6. Identified 10 most important features.
7. Built a Tableau dashboard to display geographic spread of deadly police encounters. 

**Key Findings** 

I created an [interactive map](https://public.tableau.com/profile/amanda.cheney#!/vizhome/metisproject3/map) of my findings and other select visualizations using Tableau.

Only 9% of all US zip codes account for all police killings of people of color between 2013-2020.

Top 10 Most Important Features:

|                              | Feature                                           |
| ---------------------------- | ------------------------------------------------- |
| Deadly Encounter more likely | % pop 25+ with Bachelor's degree or higher        |
|                              | % pop with High School diploma or higher          |
|                              | % of Black pop with Health Insurance              |
|                              | % of Single Mother Households with children un... |
|                              | % Latinx households with no computer              |
|                              | % of pop that is Latinx                           |
|                              | % of Black households receiving SNAP              |
|                              | % of Latinx pop with Health Insurance             |
| Deadly Encounter less likely | % all households with no computer                 |
|                              | % pop that is White (not Hispanic or Latino)      |

There are a few counter-intuitive takeaways from this analysis – namely that police killings of people of color are more likely to occur in zip codes where people of color experience poverty and utilize social safety net as well as in zip codes with higher numbers of overall population with Bachelor’s degree or higher. This may suggest the possibility of external, unobserved causes beyond zip code level socio-economic characteristics but racial dynamics ***within\*** zip codes.

Thus, where possible, future work would do well to examine characteristics of law enforcement agencies responsible for deadly encounters rather than characteristics of communities within which they exist as well as measures of racial inequality within zip codes – for instance comparing differences between median income of white and black households. Another would be to look at more granular socio-economic data, such as by census tract rather than zip codes 

**Technologies Used** 

* Jupyter Notebook
* Python
* Scikit-learn
* Pandas
* Matplotlib
* Seaborn
* YellowBrickRoad
* SQL
* Taleau 

**Classification Algorithms:** 

* KNN
* Logistic Regression
* Random Forest
* Bernoulli Naive Bayes
* SVC 

**Other**

* Classification Scoring Metrics
* Under- and Oversampling Techniques
* ROC-AUC Curves
* Precision-Recall Trade-off 
* EDA



