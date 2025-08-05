
Objective: Add Token Usage as an Analysis Dimension
Vibe Coding Tools: Gemini Code Assist
Prompts:

feature/blivezey-token-usage
Modify the application so that it also reports on token usage for each model - metrics of interest include: daily total, daily maximum, daily average per minute.
Include a token limit for each model.
Include a count of the number of times the token limit was exceeded on each day for each model. A model is said to exceed its token limit when the number of tokens per minute exceeds the token limit.
Use the total number of tokens per day per model and the cost per 1k tokens per day per model to compute a total token cost per day per model.
Do not produce a bar chart for token limit exceedances by model.
Put the token usage charts and the chart for cost per 1k tokens over time under the 'Token Usage Analysis' section and always show the charts.

blivezey/token-usage-enhancements:
Do not display the chart with weekly average cost per 1k tokens.
Display a chart chart shows cumulative costs per model.
Identify models whose number of tokens per minute is consistently well below the token limit for that model.  Suggest adjusting the limit.
Add dotted horizontal lines to the tokens per minute chart to indicate the token thresholds for each of the models.

blivezey/token-usage-enhancements:
Use the phrase 'token limit exceptions,' instead of 'token limit exceedances.'
Instead of charting daily token limit exceptions, chart monthtly token limit exceptions.
Remove the dotted horizontal lines in the tokens per minute chart that indicates the token thresholds for each of the models.
In a table, summarize the following for each model: token limit, overall token usage, overall max tokens per minute, total token limit exceptions.
Use thousands separators when displaying numeric data.

 blivezey/token-usage-wrapup:
 Add a table that shows average monthly token exceptions by cloud platform, department, project, enviroment, release version, and model.
 Add a column for department to the model token usage summary table.

 Identify model/department pairs whose number of tokens per minute is consistently well below the token limit for that model/deparment pair.  Suggest adjusting the limit to reduce cost.
 Identify model/department pairs whose number of tokens per minute frequently exceeds the token limit for that model/deparment pair.  Suggest adjusting the limit to improve response.