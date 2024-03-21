

library(DBI)
library(ggplot2)
library(RSQLite)

con <- dbConnect(SQLite(), "inference/")

mydb <- dbConnect(RSQLite::SQLite(), "")
