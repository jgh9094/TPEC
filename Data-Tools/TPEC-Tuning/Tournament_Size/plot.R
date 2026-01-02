rm(list = ls())
setwd('/Users/hernandezj45/Desktop/Repositories/GECCO-2026-TPEC/Data-Tools/TPEC-Tuning/Tournament_Size')
cat("\014")

library(ggplot2)
library(cowplot)
library(dplyr)
library(PupillometryR)
library(ggpubr)

# Themes for plots 
CONDITIONS = c(5, 10, 25, 50, 100)
SHAPE <- c(21, 21, 21, 21, 21)
cb_palette <- c('#648FFF', '#785EF0','#DC267F', '#FE6100', '#FFB000')
TSIZE <- 17
p_theme <- theme(
  plot.title = element_text(face = "bold", hjust=0.5),
  panel.border = element_blank(),
  panel.grid.minor = element_blank(),
  legend.title=element_text(size=17),
  legend.text=element_text(size=17),
  axis.title = element_text(size=17),
  axis.text = element_text(size=11),
  axis.text.y = element_text(angle = 90, hjust = 0.5),
  legend.position="bottom",
  panel.background = element_rect(fill = "#f1f2f5",
                                  colour = "white",
                                  size = 0.5, linetype = "solid")
)

# Load in all the data
df <- read.csv("./tournament_size.csv")
df$tournament_size <- factor(df$tournament_size, levels = CONDITIONS)


# Results for OpenML Task 146818
task_data <- df %>% filter(task_id == 146818)
test_data <- test_data %>% arrange(tournament_size, seed)

# Plot the testing results
ggplot(task_data, aes(x = tournament_size, y = test_accuracy, color = tournament_size, fill = tournament_size, shape = tournament_size)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Accuracy",
  ) +
  scale_x_discrete(
    name="Tournament Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette) +
  scale_fill_manual(values = cb_palette) +
  ggtitle('Accuracy on Testing Set')+
  p_theme

# Friedman test
friedman.test(test_accuracy ~ tournament_size | seed, data = task_data)

# Results for OpenML Task 359956
task_data <- df %>% filter(task_id == 359956)

# Plot the testing results
ggplot(task_data, aes(x = tournament_size, y = test_accuracy, color = tournament_size, fill = tournament_size, shape = tournament_size)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Accuracy",
  ) +
  scale_x_discrete(
    name="Tournament Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette) +
  scale_fill_manual(values = cb_palette) +
  ggtitle('Accuracy on Testing Set')+
  p_theme

# Friedman test
friedman.test(test_accuracy ~ tournament_size | seed, data = task_data)

# Wilcoxon Signed-Rank Tests between different tournament sizes
pairwise.wilcox.test(x = task_data$test_accuracy, 
                     g = task_data$tournament_size,
                     paired = TRUE,
                     p.adjust.method = "holm",
                     alternative = "less")

# Results for OpenML Task 190137
task_data <- df %>% filter(task_id == 190137)

# Plot the testing results
ggplot(task_data, aes(x = tournament_size, y = test_accuracy, color = tournament_size, fill = tournament_size, shape = tournament_size)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Accuracy",
  ) +
  scale_x_discrete(
    name="Tournament Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette) +
  scale_fill_manual(values = cb_palette) +
  ggtitle('Accuracy on Testing Set')+
  p_theme

# Friedman test
friedman.test(test_accuracy ~ tournament_size | seed, data = task_data)

# Results for OpenML Task 190411
task_data <- df %>% filter(task_id == 190411)

# Plot the testing results
ggplot(task_data, aes(x = tournament_size, y = test_accuracy, color = tournament_size, fill = tournament_size, shape = tournament_size)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Accuracy",
  ) +
  scale_x_discrete(
    name="Tournament Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette) +
  scale_fill_manual(values = cb_palette) +
  ggtitle('Accuracy on Testing Set')+
  p_theme

# Friedman test
friedman.test(test_accuracy ~ tournament_size | seed, data = task_data)

# Wilcoxon Signed-Rank Tests between different tournament sizes
pairwise.wilcox.test(x = task_data$test_accuracy, 
                     g = task_data$tournament_size,
                     paired = TRUE,
                     p.adjust.method = "holm",
                     alternative = "less")

# Results for OpenML Task 359975
task_data <- df %>% filter(task_id == 359975)

# Plot the testing results
ggplot(task_data, aes(x = tournament_size, y = test_accuracy, color = tournament_size, fill = tournament_size, shape = tournament_size)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Accuracy",
  ) +
  scale_x_discrete(
    name="Tournament Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette) +
  scale_fill_manual(values = cb_palette) +
  ggtitle('Accuracy on Testing Set')+
  p_theme

# Friedman test
friedman.test(test_accuracy ~ tournament_size | seed, data = task_data)

# Wilcoxon Signed-Rank Tests between different tournament sizes
pairwise.wilcox.test(x = task_data$test_accuracy, 
                     g = task_data$tournament_size,
                     paired = TRUE,
                     p.adjust.method = "holm",
                     alternative = "less")
