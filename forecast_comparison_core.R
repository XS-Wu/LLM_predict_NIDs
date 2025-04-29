# Forecast comparison: ARIMA, GARCH, EST, XGBoost, LSTM, and LLM with fixed random seeds

# 1. Load libraries
library(keras)
library(forecast)
library(rugarch)
library(xgboost)
library(tibble)
library(dplyr)
library(readxl)
library(stringr)
library(reticulate)
library(openxlsx)

# 2. Set global seed for reproducibility
set.seed(3407)
use_session_with_seed(
  3407,
  disable_gpu = FALSE,
  disable_parallel_cpu = FALSE,
  quiet = TRUE
)

# 3. Configure Python for Keras
# install_keras(tensorflow = "2.17.0")  # Uncomment if Keras is not installed
use_python("D:/anaconda3/python.exe", required = TRUE)
py_config()

# 4. Evaluation metrics
mae  <- function(actual, pred) { mean(abs(actual - pred), na.rm = TRUE) }
rmse <- function(actual, pred) { sqrt(mean((actual - pred)^2, na.rm = TRUE)) }
mape <- function(actual, pred) { mean(abs((actual - pred) / actual), na.rm = TRUE) * 100 }

# 5. Helper functions
# 5.1 Lag features for XGBoost
lag_data <- function(series, lags = 2) {
  df <- data.frame(y = as.numeric(series))
  for (i in seq_len(lags)) {
    df[[paste0("lag", i)]] <- dplyr::lag(df$y, n = i)
  }
  na.omit(df)
}

# 5.2 Min-max scaling and inverse
scale_data <- function(x) { (x - min(x)) / (max(x) - min(x)) }
inv_scale   <- function(x, orig) { x * (max(orig) - min(orig)) + min(orig) }

# 5.3 Sequence creation for LSTM
create_sequences <- function(data, lag = 2) {
  X <- list(); y <- list()
  for (i in (lag + 1):length(data)) {
    X[[i - lag]] <- data[(i - lag):(i - 1)]
    y[[i - lag]] <- data[i]
  }
  list(
    X = array(unlist(X), dim = c(length(X), lag, 1)),
    y = array(unlist(y), dim = c(length(y), 1))
  )
}

# 6. File path setup
data_file         <- "china_nid_2009_2025.xlsx"
llm_results_dir   <- "llm_results"
final_results_dir <- "final_results"
if (!dir.exists(final_results_dir)) dir.create(final_results_dir)

# 7. Sequential processing of LLM result files
llm_files <- list.files(llm_results_dir, pattern = "\\.xlsx$", full.names = TRUE)
for (llm_file in llm_files) {
  # 7.1 Extract disease and outcome from filename, e.g. "Influenza_cases.xlsx"
  fname <- basename(llm_file)
  parts <- strsplit(fname, "[_\\.]")[[1]]
  disease <- parts[1]
  outcome <- parts[2]
  message("Processing: ", disease, " - ", outcome)
  
  # 7.2 Read LLM predictions
  llm_data   <- read_excel(llm_file)
  llm_actual <- llm_data$Actual
  llm_pred   <- llm_data$Predicted
  
  # 7.3 Read main dataset and filter for disease & outcome
  full_data <- read_excel(data_file)
  sub_data  <- full_data %>%
    filter(.data[["指标"]] == outcome) %>%
    select(日期, all_of(disease))
  sub_data[[disease]][is.na(sub_data[[disease]])] <- 0
  sub_data$日期 <- as.Date(sub_data$日期)
  
  # 7.4 Build time series
  start_year  <- as.numeric(format(min(sub_data$日期), "%Y"))
  start_month <- as.numeric(format(min(sub_data$日期), "%m"))
  ts_data     <- ts(sub_data[[disease]], frequency = 12,
                    start = c(start_year, start_month))
  
  # 7.5 Split into train/val/test (60/20/20)
  n_total <- length(ts_data)
  n_train <- floor(0.6 * n_total)
  n_val   <- floor(0.2 * n_total)
  n_test  <- n_total - n_train - n_val
  train_ts <- window(ts_data, end = time(ts_data)[n_train])
  val_ts   <- window(ts_data, start = time(ts_data)[n_train + 1],
                     end = time(ts_data)[n_train + n_val])
  test_ts  <- window(ts_data, start = time(ts_data)[n_train + n_val + 1])
  
  # 7.6 ARIMA
  arima_fit  <- auto.arima(train_ts, stepwise = FALSE, approximation = FALSE)
  arima_fc   <- forecast(arima_fit, h = n_test)
  pred_arima <- as.numeric(arima_fc$mean)
  
  # 7.7 TGARCH
  spec_tgarch <- ugarchspec(
    variance.model = list(model = "fGARCH", submodel = "TGARCH", garchOrder = c(1,1)),
    mean.model     = list(armaOrder = c(1,0), include.mean = TRUE),
    distribution.model = "norm"
  )
  set.seed(3407)
  tgarch_fit  <- ugarchfit(spec = spec_tgarch, data = train_ts)
  tgarch_fc   <- ugarchforecast(tgarch_fit, n.ahead = n_test)
  pred_tgarch <- as.numeric(tgarch_fc@forecast$seriesFor)
  
  # 7.8 EGARCH
  spec_egarch <- ugarchspec(
    variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
    mean.model     = list(armaOrder = c(1,0), include.mean = TRUE),
    distribution.model = "norm"
  )
  set.seed(3407)
  egarch_fit  <- tryCatch(ugarchfit(spec = spec_egarch, data = train_ts, solver = "hybrid"),
                          error = function(e) NULL)
  if (!is.null(egarch_fit)) {
    egarch_fc   <- ugarchforecast(egarch_fit, n.ahead = n_test)
    pred_egarch <- as.numeric(egarch_fc@forecast$seriesFor)
  } else pred_egarch <- rep(NA, n_test)
  
  # 7.9 Exponential smoothing (ETS)
  set.seed(3407)
  ets_fit   <- ets(train_ts)
  ets_fc    <- forecast(ets_fit, h = n_test)
  pred_ets  <- as.numeric(ets_fc$mean)
  
  # 7.10 XGBoost
  set.seed(3407)
  train_df  <- lag_data(train_ts, lags = 2)
  dtrain    <- xgb.DMatrix(data = as.matrix(train_df[, -1]), label = train_df$y)
  xgb_mod   <- xgb.train(
    params = list(objective = "reg:squarederror", eta = 0.1, max_depth = 3, seed = 3407),
    data   = dtrain,
    nrounds = 100,
    verbose = 0
  )
  pred_xgb  <- numeric(n_test)
  hist_vals <- as.numeric(train_ts)
  for (i in seq_len(n_test)) {
    l1 <- tail(hist_vals, 1)
    l2 <- tail(hist_vals, 2)[1]
    pred_xgb[i] <- predict(xgb_mod, matrix(c(l1, l2), nrow = 1))
    hist_vals <- c(hist_vals, pred_xgb[i])
  }
  
  # 7.11 LSTM
  set.seed(3407)
  train_sc     <- scale_data(as.numeric(train_ts))
  seqs         <- create_sequences(train_sc, lag = 2)
  lstm_input   <- seqs$X; lstm_target <- seqs$y
  lstm_model   <- keras_model_sequential() %>%
    layer_lstm(units = 50, input_shape = c(2,1), kernel_initializer = initializer_random_uniform(seed = 3407)) %>%
    layer_dense(units = 1)
  lstm_model %>% compile(loss = "mse", optimizer = optimizer_adam())
  lstm_model %>% fit(
    x = lstm_input,
    y = lstm_target,
    epochs = 50,
    batch_size = 16,
    shuffle = FALSE,
    verbose = 0
  )
  hist_sc      <- train_sc
  pred_lstm_sc <- numeric(n_test)
  for (i in seq_len(n_test)) {
    input_seq        <- tail(hist_sc, 2)
    pred_lstm_sc[i] <- lstm_model %>% predict(array(input_seq, dim = c(1,2,1)))
    hist_sc         <- c(hist_sc, pred_lstm_sc[i])
  }
  pred_lstm <- inv_scale(pred_lstm_sc, as.numeric(train_ts))
  
}
