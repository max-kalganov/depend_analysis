library(olsrr)
library(readxl)

df <- read_excel("data/Job_prof.xls")
colnames(df) <- c("test1", "test2", "test3", "test4", "job_prof")


model <- lm(job_prof ~ test1 + test2 + test3 + test4, data = df)
k <- ols_step_all_possible(model)
k

# plot
plot(k)
