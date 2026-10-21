# REST API Documentation

Start the service with `uvicorn api.app:app --reload`. Check `GET /health`, then send `POST /predict` with a JSON object named `features` containing every training field except Attrition and the removed identifier or constant columns.

```json
{"features":{"Age":41,"BusinessTravel":"Travel_Rarely","DailyRate":1102,"Department":"Sales","DistanceFromHome":1,"Education":2,"EducationField":"Life Sciences","EnvironmentSatisfaction":2,"Gender":"Female","HourlyRate":94,"JobInvolvement":3,"JobLevel":2,"JobRole":"Sales Executive","JobSatisfaction":4,"MaritalStatus":"Single","MonthlyIncome":5993,"MonthlyRate":19479,"NumCompaniesWorked":8,"OverTime":"Yes","PercentSalaryHike":11,"PerformanceRating":3,"RelationshipSatisfaction":1,"StockOptionLevel":0,"TotalWorkingYears":8,"TrainingTimesLastYear":0,"WorkLifeBalance":1,"YearsAtCompany":6,"YearsInCurrentRole":4,"YearsSinceLastPromotion":0,"YearsWithCurrManager":5}}
```

A successful response contains `attrition_probability`, `predicted_attrition`, and `risk_category`. Missing fields return HTTP 422. The API loads the actual saved pipeline from `models/attrition_platform_model.joblib`.
