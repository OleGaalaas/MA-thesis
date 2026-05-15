library(tidyverse)
library(lubridate)
library(readr)

# ---- Load single combined dataset ----
#lenta_news_1999_2024_translated_all <- read_csv("lenta_news_1999_2024_translated_all.csv")

news_raw <- lenta_news_1999_2024_translated_all

# ---- Assign periods based on cutoff date ----
cutoff <- as.Date("2019-12-15")

news <- news_raw %>%
  mutate(
    date = as.Date(date),
    period = if_else(date <= cutoff, "1999_2019", "2019_2024")
  ) %>%
  filter(date >= as.Date("1999-01-01")) %>%
  rename(article = text_en) %>%
  select(date, title, article, period)

# ---- Basic coverage checks ----
news %>%
  summarise(
    n = n(),
    min_date = min(date, na.rm = TRUE),
    max_date = max(date, na.rm = TRUE),
    na_article = sum(is.na(article)),
    .by = period
  )

# ---- Text length ----
news <- news %>%
  mutate(text_length = nchar(article))

news %>%
  summarise(
    avg_length    = mean(text_length, na.rm = TRUE),
    median_length = median(text_length, na.rm = TRUE),
    sd_length     = sd(text_length, na.rm = TRUE),
    .by = period
  )

# ---- Article length distribution ----
ggplot(news, aes(text_length, fill = period)) +
  geom_density(alpha = 0.4) +
  scale_x_log10() +
  scale_fill_manual(
    values = c("1999_2019" = "#2166ac", "2019_2024" = "#d6604d"),
    labels = c("1999_2019" = "1999-2019", "2019_2024" = "2019-2024")
  ) +
  theme_minimal() +
  labs(
    title = "Article Length Distribution by Period",
    x     = "Article length (characters, log scale)",
    y     = "Density",
    fill  = "Period"
  )

# ---- Monthly average article length ----
monthly_length <- news %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month, period) %>%
  summarise(
    avg_length    = mean(text_length, na.rm = TRUE),
    median_length = median(text_length, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(monthly_length, aes(x = month, y = avg_length, color = period)) +
  geom_line(linewidth = 1) +
  geom_vline(
    xintercept = as.numeric(as.Date("2019-12-15")),
    linetype   = "dashed",
    color      = "black",
    linewidth  = 0.7
  ) +
  scale_color_manual(
    values = c("1999_2019" = "#2166ac", "2019_2024" = "#d6604d"),
    labels = c("1999_2019" = "1999-2019", "2019_2024" = "2019-2024")
  ) +
  theme_minimal() +
  labs(
    title  = "Average Article Length per Month",
    x      = "Month",
    y      = "Average article length (characters)",
    color  = "Period",
    caption = "Dashed line marks the December 2019 corpus cutoff point."
  )

# ---- Sentence and quote counts ----
news_feat <- news %>%
  transmute(
    date,
    period,
    month         = floor_date(date, "month"),
    text_length   = nchar(article),
    sentence_count = str_count(article, "[.!?]"),
    quote_count   = str_count(article, "\"|'")
  )

# Monthly sentences
monthly_sentences <- news_feat %>%
  group_by(month, period) %>%
  summarise(avg_sentences = mean(sentence_count, na.rm = TRUE),
            .groups = "drop")

ggplot(monthly_sentences, aes(month, avg_sentences, color = period)) +
  geom_line(linewidth = 1) +
  geom_vline(
    xintercept = as.numeric(as.Date("2019-12-15")),
    linetype   = "dashed",
    color      = "black",
    linewidth  = 0.7
  ) +
  scale_color_manual(
    values = c("1999_2019" = "#2166ac", "2019_2024" = "#d6604d"),
    labels = c("1999_2019" = "1999-2019", "2019_2024" = "2019-2024")
  ) +
  theme_minimal() +
  labs(
    title   = "Average Number of Sentences per Article per Month",
    x       = "Month",
    y       = "Sentences per article",
    color   = "Period",
    caption = "Dashed line marks the December 2019 corpus cutoff point."
  )

# Monthly quotes
monthly_quotes <- news_feat %>%
  group_by(month, period) %>%
  summarise(avg_quotes = mean(quote_count, na.rm = TRUE),
            .groups = "drop")

ggplot(monthly_quotes, aes(month, avg_quotes, color = period)) +
  geom_line(linewidth = 1) +
  geom_vline(
    xintercept = as.numeric(as.Date("2019-12-15")),
    linetype   = "dashed",
    color      = "black",
    linewidth  = 0.7
  ) +
  scale_color_manual(
    values = c("1999_2019" = "#2166ac", "2019_2024" = "#d6604d"),
    labels = c("1999_2019" = "1999-2019", "2019_2024" = "2019-2024")
  ) +
  theme_minimal() +
  labs(
    title   = "Average Number of Quotes per Article per Month",
    x       = "Month",
    y       = "Quotes per article",
    color   = "Period",
    caption = "Dashed line marks the December 2019 corpus cutoff point."
  )

# ---- Article count per month (coverage check) ----
monthly_count <- news %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month, period) %>%
  summarise(n_articles = n(), .groups = "drop")

ggplot(monthly_count, aes(month, n_articles, color = period)) +
  geom_line(linewidth = 1) +
  geom_vline(
    xintercept = as.numeric(as.Date("2019-12-15")),
    linetype   = "dashed",
    color      = "black",
    linewidth  = 0.7
  ) +
  scale_color_manual(
    values = c("1999_2019" = "#2166ac", "2019_2024" = "#d6604d"),
    labels = c("1999_2019" = "1999-2019", "2019_2024" = "2019-2024")
  ) +
  theme_minimal() +
  labs(
    title   = "Number of Articles per Month",
    x       = "Month",
    y       = "Article count",
    color   = "Period",
    caption = "Dashed line marks the December 2019 corpus cutoff point."
  )

# ---- Continuity check around cutoff ----
# Spot check: articles in the two weeks either side of the cutoff
news %>%
  filter(date >= as.Date("2019-12-01"),
         date <= as.Date("2019-12-31")) %>%
  arrange(date) %>%
  group_by(date, period) %>%
  summarise(n = n(), .groups = "drop") %>%
  print(n = 40)


#37919 jan 1 2006- jan 1 2007
#38095 jan 1 2007- jan 1 2008
#53521 jan 1 2008- jan 1 2009
#54950 jan 1 2009- jan 1 2010
#48591 jan 1 2010- jan 1 2011
#48635 jan 1 2011- jan 1 2012
#49636 jan 1 2012- jan 1 2013
#49264 jan 1 2013- jan 1 2014
#41953 jan 1 2014- jan 1 2015
#50876 an 1 2015- jan 1 2016
#70029 jan 1 2016- jan 1 2017
#64564 an 1 2017- jan 1 2018
#47795 jan 1 2018- jan 1 2019
#67894 jan 1 2019- jan 1 2020
#90734 jan 1 2020- jan 1 2021
#123650 jan 1 2021- jan 1 2022
#158138 jan 1 2022- jan 1 2023
#164861 jan 1 2023- jan 1 2024


