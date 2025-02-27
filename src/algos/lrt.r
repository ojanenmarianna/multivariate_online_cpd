# Load necessary libraries
library(ggplot2)
library(dplyr)
library(reshape2)
library(magrittr)
library(ocd)
library(gridExtra)
library(jsonlite)

# Function to handle the detection and segmentation process
detect_and_segment <- function(test_data, p, initial_offset, beta) {
  test_data <- data.matrix(test_data)

  gamma <- 24 * 60 * 60 / 1  # patience = 1 day

  # Use theoretical thresholds suggested in Chen, Wang, and Samworth (2020)
  psi <- function(t){p - 1 + t + sqrt(2 * (p - 1) * t)}
  th_diag <- log(24 * p * gamma * log2(4 * p))
  th_off_s <- 8 * log(24 * p * gamma * log2(2 * p))
  th_off_d <- psi(th_off_s / 4)
  thresh <- setNames(c(th_diag, th_off_d, th_off_s),
                     c("diag", "off_d", "off_s"))

  changepoints <- c()
  current_test_data <- test_data
  current_offset <- initial_offset

  repeat {
    segment_length <- nrow(current_test_data)

    # Ensure there's enough data for training
    if (segment_length <= 200) {
      cat("Not enough data for training in current segment.\n")
      break
    }

    train <- current_test_data[1:100, ]
    test <- current_test_data[101:segment_length, ]

    # Initialize the detector for each segment
    detector <- ChangepointDetector(dim = p,
                                    method = "ocd",
                                    thresh = thresh,
                                    beta = beta)
    detector %<>% setStatus("estimating")
    for (i in 1:nrow(train)) {
      detector %<>% getData(train[i, ])
    }

    detector %<>% setStatus("monitoring")

    change_detected <- FALSE
    for (i in 1:nrow(test)) {
      detector %<>% getData(test[i, ])
      if (is.numeric(detector %>% status)) {
        # Offset by train size + current test index
        time_declared <- current_offset + i + 100
        changepoints <- c(changepoints, time_declared)
        change_detected <- TRUE
        break
      }
    }

    if (!change_detected) {
      break
    }

    # Segment the data from the detected changepoint and continue
    new_start_index <- which(as.numeric(rownames(test_data)) > time_declared)[1]
    if (is.na(new_start_index) || new_start_index > nrow(test_data)) {
      break
    }

    current_test_data <- test_data[new_start_index:nrow(test_data), ]
    current_offset <- time_declared
  }

  return(changepoints)
}
