#!/usr/bin/env Rscript

# load the necessary packages
source("psycholinguistic_analyses/packages.R")

# clear all variables
rm(list = ls())

set.seed(123)
theme_set(theme_light())
options(digits = 8)
options(dplyr.summarise.inform = TRUE)

# Load the YAML file
config <- yaml.load_file("psycholinguistic_analyses/CONSTANTS_ANALYSES.yaml")


z_score <- function(x) {
    return((x - mean(x)) / sd(x))
}

z_score_test <- function(x, sd) {
    return((x - mean(x)) / sd)
}

remove_outlier <- function(df, reading_measure) {
    reading_times <- as.numeric(df[[reading_measure]])
    z_score <- z_score(reading_times)
    abs_z_score <- abs(z_score)
    df$outlier <- abs_z_score > 3
    # print number of outliers / total number of reading times
    print(paste(sum(df$outlier), "/", length(df$outlier)))
    # remove outliers
    df <- df[df$outlier == FALSE, ]
    return(df)
}

preprocess <- function(df, predictors_to_normalize, is_linear) {
    # first, copy df in order to not overwrite original
    df_copy <- df
    #df_copy$subj_id <- as.factor(df_copy$subject_id)
    #  convert to log lex freq
    df_copy$log_lex_freq <- as.numeric(df_copy$zipf_freq)

    # normalize baseline predictors
    df_copy$log_lex_freq <- scale(df_copy$log_lex_freq)
    df_copy$word_len <- scale(df_copy$word_len)

    # normalize surprisal/entropy predictors
    for (predictor in predictors_to_normalize) {
        df_copy[[predictor]] <- as.numeric(df_copy[[predictor]])
        df_copy[[predictor]] <- scale(df_copy[[predictor]])
    }
    return(df_copy)
}

is_significant <- function(p_value, alpha = 0.05) {
    ifelse(p_value < alpha, "sig.", "not sig.")
}



# Prepare experiments 
# log-transform response variable?
LOG_TF <- TRUE

# continuous response variables 
CONT_RESP_VARIABLES <- c("FFD", "SFD", "FD", "FPRT", "TFT", "RRT", "RPD_inc")

rms_df = read.csv("psycholinguistic_analyses/reading_measures_annotated.csv", header = TRUE, sep = "\t")

reading_measures <- c("FPRT", "TFT", "RRT", "RPD_inc", "Fix", "RR")

PREDICTORS_TO_NORMALIZE <- c("surprisal")


#models <- c("human", "swift", "ez-reader", "scandl-fixdur")
models <- c("human", "scandl-fixdur", "ez-reader", "swift")
settings <- c("reader", "sentence", "combined", "cross_dataset")



# Initialize an empty list to store results
results <- list()

for (model in models) {

    for (setting in settings) {

        for (reading_measure in reading_measures) {

            # Subset the data frame to the current model and setting 
            df <- rms_df[rms_df$model == model & rms_df$setting == setting, ]
            is_linear <- TRUE
            
            if (reading_measure %in% CONT_RESP_VARIABLES) {
                if (LOG_TF) {
                    # Remove 0s 
                    df <- df[df[[reading_measure]] != 0, ]
                    # Log-transform the response variable
                    df[[reading_measure]] <- log(df[[reading_measure]])
                }
            } else {
                is_linear <- FALSE
            }
            
            # Preprocess the predictors 
            df <- preprocess(df, PREDICTORS_TO_NORMALIZE, is_linear)
            mixed_effects <- FALSE
            
            if (model == "human") {
                formula_model <- paste(reading_measure, "~ 1 + (1|reader_id) + word_len + log_lex_freq + surprisal")
                mixed_effects <- TRUE
            } else {
                formula_model <- paste(reading_measure, "~ 1 + word_len + log_lex_freq + surprisal")
            }
            
            # Fit the model
            if (mixed_effects) {
                if (is_linear) {
                    print(paste("Running lmer for", model, setting, reading_measure))
                    linear_model <- lmer(as.formula(formula_model), data = df)
                } else {
                    print(paste("Running glmer for", model, setting, reading_measure))
                    linear_model <- glmer(as.formula(formula_model), data = df, family = "binomial")
                }
            } else {
                if (is_linear) {
                    print(paste("Running lm for", model, setting, reading_measure))
                    linear_model <- lm(as.formula(formula_model), data = df)
                } else {
                    print(paste("Running glm for", model, setting, reading_measure))
                    linear_model <- glm(as.formula(formula_model), data = df, family = "binomial")
                }
            }
            
            # Extract coefficients and other statistics
            summary_model <- summary(linear_model)
            coefs <- summary_model$coefficients

            if (!is_linear) {
                statistic_name <- "z value"
                p_value_name <- "Pr(>|z|)"
            } else {
                statistic_name <- "t value"
                p_value_name <- "Pr(>|t|)"
            }
            
            # Store results in a list
            for (i in 1:nrow(coefs)) {
                results[[length(results) + 1]] <- data.frame(
                    model = model,
                    setting = setting,
                    reading_measure = reading_measure,
                    term = rownames(coefs)[i],
                    estimate = coefs[i, "Estimate"],
                    std_error = coefs[i, "Std. Error"],
                    t_value = coefs[i, statistic_name],
                    p_value = coefs[i, p_value_name]  # Adjust if using glm (use "Pr(>|z|)")
                )
            }
        }
    }
}

# Combine results into a single data frame
results_df <- do.call(rbind, results)

# create two new columns: one "ci_lower" and one "ci_upper"
results_df$ci_lower <- results_df$estimate - (1.96 * results_df$std_error)
results_df$ci_upper <- results_df$estimate + (1.96 * results_df$std_error)

if (!dir.exists("psycholinguistic_analyses/results")) {
    dir.create("psycholinguistic_analyses/results")
} else {
    print("Directory already exists")
}

# save results as csv file with tab separator
write.table(results_df, file = "psycholinguistic_analyses/results/results.csv", sep = "\t", row.names = FALSE)




results_df_plots <- results_df %>%
  filter(term != "(Intercept)")
results_df_plots  <- results_df_plots %>%
  mutate(term = recode(term,
                       log_lex_freq = "lexical freq.",
                       word_len = "word length"))

results_df_plots <- results_df_plots %>%
  mutate(reading_measure = recode(reading_measure,
                                  'TFT' = 'total-fixation time',
                                  'RRT' = 're-reading time',
                                  'FPRT' = 'first-pass reading time',
                                  'RR' = 're-reading',
                                  'RPD_inc' = 'go-past time',
                                  'Fix' = 'fixated'))



plot_results_setting <- function(results_df, setting) {
    if (!dir.exists("psycholinguistic_analyses/results/setting")) {
        dir.create("psycholinguistic_analyses/results/setting")
    }
    # Define custom colors and labels
    custom_colors <- c("scandl-fixdur" = "#E41A1C", "human" = "#377EB8", "ez-reader" = "#4DAF4A", "swift" = "#FFA500")  
    custom_labels <- c("scandl-fixdur" = "ScanDL 2.0", "human" = "Human", "ez-reader" = "E-Z Reader", "swift" = "SWIFT") 

    # Set a minimal theme and apply custom font sizes
    theme_set(theme_light(base_size = 14))  # Set a base font size

    p <- ggplot(results_df[results_df$setting == setting, ], aes(x = term, y = estimate, color = model, fill = model)) +
        geom_point(position = position_dodge(width = 0.5), size = 3) +
        geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), position = position_dodge(width = 0.5), width = 0.25) +
        facet_wrap(~reading_measure, scales = "free") +
        labs(x = "Predictor", y = "Estimate (log RTs)") +
        scale_color_manual(values = custom_colors, labels = custom_labels) +
        scale_fill_manual(values = custom_colors, labels = custom_labels) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 14),     # X-axis text size
            axis.text.y = element_text(size = 14),                            # Y-axis text size
            axis.title.x = element_text(size = 16),                           # X-axis label size
            axis.title.y = element_text(size = 16),                           # Y-axis label size
            strip.text = element_text(size = 16),                             # Facet wrap label size
            legend.text = element_text(size = 14),                            # Legend text size
            legend.title = element_text(size = 14),                           # Legend title size
            legend.position = "top"
        )
    
    ggsave(paste0("psycholinguistic_analyses/results/setting/", setting, ".png"), plot = p, width = 10, height = 16, dpi = 300)
}

for (setting in settings) {
    plot_results_setting(results_df_plots, setting)
}


# function to create plots for one reading measure
# setting is the facet wrap, model is the color
# surprisal, word length, and lexical frequency are the x-axis
# the y-axis is the estimate, and the error bars are the confidence intervals



plot_results_reading_measure <- function(results_df, reading_measure) {
    if (!dir.exists("psycholinguistic_analyses/results/reading_measure")) {
        dir.create("psycholinguistic_analyses/results/reading_measure")
    }
    # Define custom colors and labels
    custom_colors <- c("scandl-fixdur" = "#E41A1C", "human" = "#377EB8", "ez-reader" = "#4DAF4A", "swift" = "#FFA500")  
    custom_labels <- c("scandl-fixdur" = "ScanDL 2.0", "human" = "Human", "ez-reader" = "E-Z Reader", "swift" = "SWIFT") 

    # Set a minimal theme and apply custom font sizes
    theme_set(theme_light(base_size = 14))  # Set a base font size

    p <- ggplot(results_df[results_df$reading_measure == reading_measure, ], aes(x = term, y = estimate, color = model, fill = model)) +
        geom_point(position = position_dodge(width = 0.5), size = 3) +
        geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), position = position_dodge(width = 0.5), width = 0.25) +
        facet_wrap(~setting, scales = "free") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(x = "Predictor", y = "Estimate (log RTs)") +   # title = paste("Results for", reading_measure), 
        theme(legend.position = "top") +
        scale_color_manual(values = custom_colors, labels = custom_labels) +  # Apply custom colors and labels to points
        scale_fill_manual(values = custom_colors, labels = custom_labels)      # Apply custom colors and labels to fills
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 14),     # X-axis text size
            axis.text.y = element_text(size = 14),                            # Y-axis text size
            axis.title.x = element_text(size = 16),                           # X-axis label size
            axis.title.y = element_text(size = 16),                           # Y-axis label size
            strip.text = element_text(size = 16),                             # Facet wrap label size
            legend.text = element_text(size = 14),                            # Legend text size
            legend.title = element_text(size = 14),                           # Legend title size
            legend.position = "top"
        )
    
    filename_reading_measure <- gsub(' ', '_', reading_measure)
    ggsave(paste0("psycholinguistic_analyses/results/reading_measure/", filename_reading_measure, ".png"), plot = p, width = 10, height = 16, dpi = 300)
}


reading_measures_renamed <- unique(results_df_plots$reading_measure)
for (reading_measure in reading_measures_renamed) {
    plot_results_reading_measure(results_df_plots, reading_measure)
}
