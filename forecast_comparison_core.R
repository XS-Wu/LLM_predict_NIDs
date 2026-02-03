# Forecast comparison: ARIMA, GARCH, EST, XGBoost, LSTM, and LLM with fixed random seeds

options(stringsAsFactors = FALSE)
Sys.setenv("TF_CPP_MIN_LOG_LEVEL" = "2")
set.seed(3407)

# -------------------- 0) Small utils --------------------
`%||%` <- function(a, b) if (!is.null(a)) a else b

sanitize_filename <- function(x, max_len = 180) {
  x <- gsub("[<>:\"/\\\\|?*]+", "_", x)
  x <- gsub("\\s+", " ", x)
  x <- trimws(x)
  if (nchar(x) > max_len) x <- substr(x, 1, max_len)
  x
}

align_len <- function(x, n) {
  x <- as.numeric(x)
  if (length(x) == n) return(x)
  if (length(x) > n) return(x[1:n])
  c(x, rep(NA_real_, n - length(x)))
}

mae <- function(actual, pred) {
  ok <- is.finite(actual) & is.finite(pred)
  if (!any(ok)) return(NA_real_)
  mean(abs(actual[ok] - pred[ok]))
}
rmse <- function(actual, pred) {
  ok <- is.finite(actual) & is.finite(pred)
  if (!any(ok)) return(NA_real_)
  sqrt(mean((actual[ok] - pred[ok])^2))
}
mape <- function(actual, pred) {
  ok <- is.finite(actual) & is.finite(pred) & actual != 0
  if (!any(ok)) return(NA_real_)
  mean(abs((actual[ok] - pred[ok]) / actual[ok])) * 100
}
mse <- function(actual, pred) {
  ok <- is.finite(actual) & is.finite(pred)
  if (!any(ok)) return(NA_real_)
  mean((actual[ok] - pred[ok])^2)
}

# Fixed 6:2:2 split 
fixed_split_622 <- function(n_total) {
  n_train <- floor(0.6 * n_total)
  n_val   <- floor(0.2 * n_total)
  n_test  <- n_total - n_train - n_val
  if (n_train <= 0 || n_val <= 0 || n_test <= 0) {
    stop("Series too short for 6:2:2 split: n_total=", n_total,
         " -> train=", n_train, " val=", n_val, " test=", n_test)
  }
  list(n_train=n_train, n_val=n_val, n_test=n_test, scheme="fixed_floor_6_2_2")
}

# -------------------- 0.1) GARCH tuning helpers (VAL select; TEST forecast from train) --------------------
garch_grid <- expand.grid(
  arma_p  = 0,
  arma_q  = 0,
  garch_p = 1,
  garch_q = 1,
  dist    = c("norm","std"),
  stringsAsFactors = FALSE
)

garch_forecast <- function(train_ts, h, type=c("TGARCH","EGARCH"), par_row) {
  type <- match.arg(type)
  par_row <- as.list(par_row)
  
  spec <- if (type == "TGARCH") {
    rugarch::ugarchspec(
      variance.model = list(model="fGARCH", submodel="TGARCH",
                            garchOrder=c(par_row$garch_p, par_row$garch_q)),
      mean.model = list(armaOrder=c(par_row$arma_p, par_row$arma_q), include.mean=TRUE),
      distribution.model = par_row$dist
    )
  } else {
    rugarch::ugarchspec(
      variance.model = list(model="eGARCH",
                            garchOrder=c(par_row$garch_p, par_row$garch_q)),
      mean.model = list(armaOrder=c(par_row$arma_p, par_row$arma_q), include.mean=TRUE),
      distribution.model = par_row$dist
    )
  }
  
  fit <- suppressMessages(suppressWarnings(
    rugarch::ugarchfit(spec=spec, data=train_ts, solver="hybrid", fit.control=list(scale=1))
  ))
  
  if (fit@fit$convergence != 0) stop("GARCH not converged")
  cf <- coef(fit)
  if (length(cf) == 0 || any(!is.finite(cf))) stop("Bad coefficients")
  
  as.numeric(rugarch::ugarchforecast(fit, n.ahead=h)@forecast$seriesFor)
}

tune_garch <- function(train_ts, val_ts, type=c("TGARCH","EGARCH"), grid=garch_grid) {
  type <- match.arg(type)
  best <- list(score=Inf, par=NULL, val_pred=rep(NA_real_, length(val_ts)))
  yv <- as.numeric(val_ts)
  
  for (i in seq_len(nrow(grid))) {
    pr <- grid[i,]
    vp <- tryCatch(garch_forecast(train_ts, length(val_ts), type=type, par_row=pr),
                   error=function(e) NULL)
    if (is.null(vp)) next
    sc <- mse(yv, vp)
    
    if (is.finite(sc) && sc < best$score) {
      best <- list(score=sc, par=pr, val_pred=vp)
    }
  }
  best
}

par_to_str <- function(par_df) {
  if (is.null(par_df) || nrow(par_df) == 0) return(NA_character_)
  pr <- as.list(par_df[1,])
  paste(names(pr), unlist(pr), sep="=", collapse=";")
}

# -------------------- 1) Packages --------------------
pkgs <- c("keras3","forecast","rugarch","xgboost","tibble","dplyr","readxl","reticulate","openxlsx")
to_install <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(keras3)
  library(forecast)
  library(rugarch)
  library(xgboost)
  library(tibble)
  library(dplyr)
  library(readxl)
  library(reticulate)
  library(openxlsx)
})

clear_keras_session <- function() {
  try(keras3::clear_session(), silent = TRUE)
}

# -------------------- 2) Python / TensorFlow (for keras3) --------------------
# PYTHON_PATH <- "path/to/python"
# use_python(PYTHON_PATH, required = TRUE)

py_config()
if (!py_module_available("tensorflow")) {
  stop("TensorFlow was not found in the active Python environment.")
}
try(py_run_string("import tensorflow as tf; tf.random.set_seed(3407)"), silent = TRUE)

# -------------------- 3) Inputs --------------------
# Main data must be a long table with the following columns:
# dataset, outcome, country, disease, group, date, output
# - date: "YYYY-MM-01" style is recommended; will be converted to Date
# - output: numeric

MAIN_DATA_PATH <- "path/to/your_long_data.csv"

# LLM prediction files:
# - One .xlsx per series
# - Must contain a numeric column named "predicted" (or "prediction")
# - Optional: "date" column (Date/POSIXct/excel numeric/ISO string)
# - The file name must include tokens that allow matching:
#   dataset tag (e.g., "CHINA" or "USAUS"), outcome tag ("incidence" or "death"),
#   plus disease (and for CHINA, group is strongly recommended).
LLM_PRED_DIR <- "path/to/llm_predictions_folder"

OUTPUT_DIR <- "path/to/output_folder"
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# -------------------- 4) Load main data (no reshaping here) --------------------
read_main_data <- function(fp) {
  df <- tryCatch(read.csv(fp, stringsAsFactors = FALSE), error=function(e) NULL)
  if (is.null(df) || nrow(df) == 0) stop("Failed to read main data: ", fp)
  
  need <- c("dataset","outcome","country","disease","group","date","output")
  miss <- setdiff(need, tolower(names(df)))
  if (length(miss) > 0) stop("Main data is missing columns: ", paste(miss, collapse=", "))
  
  names(df) <- tolower(names(df))
  
  df$date <- as.Date(df$date)
  if (any(is.na(df$date))) stop("Invalid date values found in main data (expected Date or ISO date string).")
  
  df$output <- suppressWarnings(as.numeric(df$output))
  df$output <- ifelse(is.na(df$output), 0, df$output)
  
  df %>%
    transmute(
      dataset = as.character(dataset),
      outcome = as.character(outcome),
      country = as.character(country),
      disease = as.character(disease),
      group   = as.character(group),
      date    = as.Date(date),
      output  = as.numeric(output)
    ) %>%
    arrange(dataset, outcome, country, disease, group, date)
}

full_data <- read_main_data(MAIN_DATA_PATH)

# -------------------- 5) LLM file catalog + reader --------------------
infer_dataset_from_filename <- function(fn) {
  if (grepl("CHINA", fn, ignore.case = TRUE)) return("CHINA")
  if (grepl("USAUS", fn, ignore.case = TRUE)) return("USAUS")
  stop("Cannot infer dataset from LLM filename: ", fn)
}

infer_outcome_from_filename <- function(fn) {
  if (grepl("death", fn, ignore.case = TRUE)) return("death")
  if (grepl("incidence|case", fn, ignore.case = TRUE)) return("incidence")
  stop("Cannot infer outcome from LLM filename (use 'incidence' or 'death'): ", fn)
}

coerce_atomic1 <- function(x, mode=c("chr","num","date")) {
  mode <- match.arg(mode)
  if (is.list(x)) {
    if (mode == "chr") return(vapply(x, function(z) { z <- unlist(z); if (length(z)==0) NA_character_ else as.character(z[1]) }, ""))
    if (mode == "num") return(vapply(x, function(z) { z <- suppressWarnings(as.numeric(unlist(z))); if (length(z)==0) NA_real_ else z[1] }, numeric(1)))
  }
  if (mode == "chr") return(as.character(x))
  if (mode == "num") return(suppressWarnings(as.numeric(x)))
  return(x)
}

.llm_cache <- new.env(parent = emptyenv())
.llm_used  <- new.env(parent = emptyenv())

read_llm_file <- function(fp) {
  if (exists(fp, envir = .llm_cache, inherits = FALSE)) return(get(fp, envir = .llm_cache))
  
  df <- tryCatch(read_excel(fp), error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) {
    assign(fp, NULL, envir = .llm_cache)
    return(NULL)
  }
  names(df) <- tolower(names(df))
  
  if (!("predicted" %in% names(df)) && ("prediction" %in% names(df))) df$predicted <- df$prediction
  if (!("predicted" %in% names(df))) {
    assign(fp, NULL, envir = .llm_cache)
    return(NULL)
  }
  
  pred <- coerce_atomic1(df$predicted, "num")
  
  actual_col <- names(df)[grepl("^actual$|^y$|y_true|label|output", names(df))][1] %||% NA_character_
  actual <- if (!is.na(actual_col)) coerce_atomic1(df[[actual_col]], "num") else rep(NA_real_, length(pred))
  
  date_vec <- rep(as.Date(NA), length(pred))
  if ("date" %in% names(df)) {
    dv <- df$date
    if (inherits(dv, "POSIXct") || inherits(dv, "POSIXt")) {
      date_vec <- as.Date(format(as.Date(dv), "%Y-%m-01"))
    } else if (inherits(dv, "Date")) {
      date_vec <- as.Date(format(dv, "%Y-%m-01"))
    } else if (is.numeric(dv)) {
      date_vec <- as.Date(format(as.Date(as.numeric(dv), origin="1899-12-30"), "%Y-%m-01"))
    } else {
      tmp <- suppressWarnings(as.Date(as.character(dv)))
      if (all(!is.na(tmp))) date_vec <- as.Date(format(tmp, "%Y-%m-01"))
    }
  }
  
  out <- tibble(
    predicted = as.numeric(pred),
    actual    = as.numeric(actual),
    date      = as.Date(date_vec)
  )
  
  assign(fp, out, envir = .llm_cache)
  out
}

canon <- function(x) {
  x <- as.character(x)
  x <- tolower(x)
  gsub("[^a-z0-9]+", "", x)
}

name_score <- function(filename, country_i, disease_i, group_i) {
  s <- 0L
  if (!is.na(disease_i) && nzchar(disease_i)) s <- s + 5L * as.integer(grepl(disease_i, filename, fixed = TRUE))
  if (!is.na(group_i)   && nzchar(group_i))   s <- s + 3L * as.integer(grepl(group_i, filename, fixed = TRUE))
  if (!is.na(country_i) && nzchar(country_i)) s <- s + 2L * as.integer(grepl(country_i, filename, fixed = TRUE))
  s
}

llm_files <- list.files(LLM_PRED_DIR, pattern = "\\.xlsx$", full.names = TRUE)
if (length(llm_files) == 0) stop("No .xlsx found in LLM_PRED_DIR: ", LLM_PRED_DIR)

llm_catalog <- tibble(
  filepath = llm_files,
  filename = basename(llm_files),
  dataset  = vapply(basename(llm_files), infer_dataset_from_filename, character(1)),
  outcome  = vapply(basename(llm_files), infer_outcome_from_filename, character(1))
)

select_llm_file_for_series_testonly <- function(dataset_i, outcome_i, country_i, disease_i, group_i,
                                                df_test, top_k = 50) {
  cand <- llm_catalog %>% dplyr::filter(dataset == dataset_i, outcome == outcome_i)
  if (nrow(cand) == 0) return(list(fp=NA_character_, reason="no_llm_candidates"))
  
  used_flag <- vapply(cand$filepath, function(p) exists(p, envir=.llm_used, inherits=FALSE), logical(1))
  cand <- cand[!used_flag, , drop=FALSE]
  if (nrow(cand) == 0) return(list(fp=NA_character_, reason="all_llm_files_already_used"))
  
  fn_c  <- canon(cand$filename)
  dis_c <- canon(disease_i)
  
  if (is.na(dis_c) || !nzchar(dis_c)) return(list(fp=NA_character_, reason="empty_disease_key"))
  keep <- grepl(dis_c, fn_c, fixed = TRUE)
  
  if (dataset_i == "CHINA") {
    grp_c <- canon(group_i)
    if (is.na(grp_c) || !nzchar(grp_c)) return(list(fp=NA_character_, reason="empty_group_key"))
    keep <- keep & grepl(grp_c, fn_c, fixed = TRUE)
  }
  
  cand <- cand[keep, , drop=FALSE]
  if (nrow(cand) == 0) return(list(fp=NA_character_, reason="no_strict_name_match"))
  
  cand <- cand %>%
    dplyr::mutate(ns = vapply(filename, name_score, integer(1),
                              country_i=country_i, disease_i=disease_i, group_i=group_i)) %>%
    dplyr::arrange(desc(ns), desc(nchar(filename))) %>%
    dplyr::slice_head(n = min(top_k, nrow(cand)))
  
  n_test <- nrow(df_test)
  y_ref  <- suppressWarnings(as.numeric(df_test$output))
  
  best <- list(fp=NA_character_, ns=-1L, check_mae=Inf)
  
  for (i in seq_len(nrow(cand))) {
    fp <- cand$filepath[i]
    tbl <- read_llm_file(fp)
    if (is.null(tbl) || nrow(tbl) == 0) next
    if (nrow(tbl) != n_test) next
    
    if (any(!is.na(tbl$date))) {
      if (!identical(as.Date(tbl$date), as.Date(df_test$date))) next
    }
    
    check_vec <- if (any(is.finite(tbl$actual))) tbl$actual else tbl$predicted
    ok <- is.finite(check_vec) & is.finite(y_ref)
    check_mae <- if (any(ok)) mean(abs(check_vec[ok] - y_ref[ok])) else Inf
    
    if (is.na(best$fp) ||
        cand$ns[i] > best$ns ||
        (cand$ns[i] == best$ns && check_mae < best$check_mae)) {
      best <- list(fp=fp, ns=cand$ns[i], check_mae=check_mae)
    }
  }
  
  if (is.na(best$fp)) return(list(fp=NA_character_, reason="no_length_or_date_match_after_strict_name"))
  list(fp=best$fp, ns=best$ns, check_mae=best$check_mae, reason="ok")
}

# -------------------- 6) ML helpers (XGB/LSTM) --------------------
make_lag_xy <- function(y, lags, t_start, t_end) {
  n <- length(y)
  if (t_start <= lags) stop("t_start must be > lags")
  if (t_end > n) stop("t_end exceeds series length")
  idx <- t_start:t_end
  X <- matrix(NA_real_, nrow = length(idx), ncol = lags)
  for (j in 1:lags) X[, j] <- y[idx - j]
  colnames(X) <- paste0("lag", 1:lags)
  list(X = X, y = y[idx])
}

recursive_forecast_xgb <- function(model, history, n_ahead, lags) {
  preds <- numeric(n_ahead)
  hist <- as.numeric(history)
  for (i in 1:n_ahead) {
    lagk <- hist[(length(hist)-lags+1):length(hist)]
    x <- matrix(rev(lagk), nrow=1)
    colnames(x) <- paste0("lag", 1:lags)
    p <- as.numeric(predict(model, x))
    preds[i] <- p
    hist <- c(hist, p)
  }
  preds
}

get_rng <- function(x) range(x, na.rm = TRUE)
scale_with_rng <- function(x, rng) {
  if (isTRUE(all.equal(rng[1], rng[2]))) return(rep(0, length(x)))
  (x - rng[1]) / (rng[2] - rng[1])
}
inv_scale_with_rng <- function(x, rng) {
  if (isTRUE(all.equal(rng[1], rng[2]))) return(rep(rng[1], length(x)))
  x * (rng[2] - rng[1]) + rng[1]
}

create_sequences_indexed <- function(vec_scaled, lag = 2) {
  n <- length(vec_scaled)
  if (n <= lag) return(NULL)
  X <- vector("list", n - lag)
  y <- vector("list", n - lag)
  idx_target <- integer(n - lag)
  for (i in (lag + 1):n) {
    X[[i - lag]] <- vec_scaled[(i - lag):(i - 1)]
    y[[i - lag]] <- vec_scaled[i]
    idx_target[i - lag] <- i
  }
  X_array <- array(unlist(X), dim = c(length(X), lag, 1))
  y_array <- array(unlist(y), dim = c(length(y), 1))
  list(X = X_array, y = y_array, idx = idx_target)
}

recursive_forecast_lstm <- function(model, history_scaled, n_ahead, lags) {
  preds <- numeric(n_ahead)
  hist <- as.numeric(history_scaled)
  for (i in 1:n_ahead) {
    input_seq <- tail(hist, lags)
    input_arr <- array(input_seq, dim = c(1, lags, 1))
    p <- as.numeric(model$predict(input_arr, verbose = 0))
    preds[i] <- p
    hist <- c(hist, p)
  }
  preds
}

# -------------------- 7) Series keys --------------------
series_keys <- full_data %>% distinct(dataset, outcome, country, disease, group)
message("Number of series: ", nrow(series_keys))

# -------------------- 8) Run one series (TEST-only output) --------------------
run_one_series <- function(dataset_i, outcome_i, country_i, disease_i, group_i,
                           lags = 2,
                           xgb_max_rounds = 800, xgb_early_stop = 30,
                           lstm_units = 50, lstm_max_epochs = 80, lstm_patience = 10, lstm_batch = 16) {
  
  df <- full_data %>%
    filter(dataset == dataset_i, outcome == outcome_i,
           country == country_i, disease == disease_i, group == group_i) %>%
    arrange(date)
  
  n_total <- nrow(df)
  sp <- fixed_split_622(n_total)
  n_train <- sp$n_train; n_val <- sp$n_val; n_test <- sp$n_test
  split_scheme <- sp$scheme
  
  df_train <- df[1:n_train, , drop=FALSE]
  df_val   <- df[(n_train+1):(n_train+n_val), , drop=FALSE]
  df_test  <- df[(n_train+n_val+1):n_total, , drop=FALSE]
  
  sel <- select_llm_file_for_series_testonly(
    dataset_i, outcome_i, country_i, disease_i, group_i,
    df_test = df_test
  )
  if (is.na(sel$fp)) {
    stop("No matching LLM file for this series (n_test match required). reason=", sel$reason %||% "NA")
  }
  llm_tbl <- read_llm_file(sel$fp)
  assign(sel$fp, TRUE, envir = .llm_used)
  
  if (any(!is.na(llm_tbl$date)) && !identical(as.Date(llm_tbl$date), as.Date(df_test$date))) {
    stop("LLM TEST dates do not match baseline TEST dates: ", basename(sel$fp))
  }
  if (is.null(llm_tbl) || nrow(llm_tbl) == 0) stop("LLM file is empty: ", basename(sel$fp))
  if (nrow(llm_tbl) != n_test) stop("LLM rows != n_test: ", basename(sel$fp))
  
  llm_pred_test <- as.numeric(llm_tbl$predicted)
  if (any(!is.finite(llm_pred_test))) stop("Non-finite values in LLM predicted: ", basename(sel$fp))
  
  start_y <- as.integer(format(df$date[1], "%Y"))
  start_m <- as.integer(format(df$date[1], "%m"))
  
  train     <- ts(df_train$output, frequency=12, start=c(start_y, start_m))
  train_val <- ts(c(df_train$output, df_val$output), frequency=12, start=c(start_y, start_m))
  val       <- ts(df_val$output, frequency=12,
                  start=c(as.integer(format(df_val$date[1], "%Y")), as.integer(format(df_val$date[1], "%m"))))
  test      <- ts(df_test$output, frequency=12,
                  start=c(as.integer(format(df_test$date[1], "%Y")), as.integer(format(df_test$date[1], "%m"))))
  
  # -------------------- ARIMA (VAL select by MSE; TEST forecast from train) --------------------
  arima_candidates <- list(
    list(name="A1_full",   stepwise=FALSE, approximation=FALSE, seasonal=TRUE),
    list(name="A2_fast",   stepwise=TRUE,  approximation=FALSE, seasonal=TRUE),
    list(name="A3_faster", stepwise=TRUE,  approximation=TRUE,  seasonal=TRUE),
    list(name="A4_noseas", stepwise=TRUE,  approximation=TRUE,  seasonal=FALSE)
  )
  
  best_arima_name <- NA_character_
  best_arima_val_mse <- Inf
  yv <- as.numeric(val)
  
  for (cand in arima_candidates) {
    vp <- tryCatch({
      fit <- forecast::auto.arima(train, stepwise=cand$stepwise,
                                  approximation=cand$approximation, seasonal=cand$seasonal)
      as.numeric(forecast::forecast(fit, h = n_val)$mean)
    }, error=function(e) NULL)
    
    if (!is.null(vp)) {
      vmse <- mse(yv, vp)
      if (is.finite(vmse) && vmse < best_arima_val_mse) {
        best_arima_val_mse <- vmse
        best_arima_name <- cand$name
      }
    }
  }
  
  pred_arima_test <- tryCatch({
    cand <- arima_candidates[[which(vapply(arima_candidates, function(x) x$name==best_arima_name, logical(1)))[1]]]
    fit_tr <- forecast::auto.arima(train, stepwise=cand$stepwise,
                                   approximation=cand$approximation, seasonal=cand$seasonal)
    all_fc <- as.numeric(forecast::forecast(fit_tr, h = n_val + n_test)$mean)
    tail(all_fc, n_test)
  }, error=function(e) rep(NA_real_, n_test))
  
  # -------------------- TGARCH / EGARCH (VAL tuning; TEST forecast from train) --------------------
  best_tg <- tryCatch(tune_garch(train, val, type="TGARCH"), error=function(e) NULL)
  pred_tgarch_test <- tryCatch({
    if (is.null(best_tg) || is.null(best_tg$par)) stop("No valid TGARCH model")
    all_fc <- garch_forecast(train, n_val + n_test, type="TGARCH", par_row=best_tg$par)
    align_len(tail(all_fc, n_test), n_test)
  }, error=function(e) rep(NA_real_, n_test))
  tgarch_best_str <- if (!is.null(best_tg) && !is.null(best_tg$par)) par_to_str(best_tg$par) else NA_character_
  
  best_eg <- tryCatch(tune_garch(train, val, type="EGARCH"), error=function(e) NULL)
  pred_egarch_test <- tryCatch({
    if (is.null(best_eg) || is.null(best_eg$par)) stop("No valid EGARCH model")
    all_fc <- garch_forecast(train, n_val + n_test, type="EGARCH", par_row=best_eg$par)
    align_len(tail(all_fc, n_test), n_test)
  }, error=function(e) rep(NA_real_, n_test))
  egarch_best_str <- if (!is.null(best_eg) && !is.null(best_eg$par)) par_to_str(best_eg$par) else NA_character_
  
  # -------------------- ETS (VAL select by MSE; TEST forecast from train) --------------------
  ets_candidates <- list(
    list(name="E1_auto",   damped=FALSE),
    list(name="E2_damped", damped=TRUE)
  )
  best_ets_name <- NA_character_
  best_ets_val_mse <- Inf
  yv <- as.numeric(val)
  
  for (cand in ets_candidates) {
    vp <- tryCatch({
      fit <- forecast::ets(train, damped=cand$damped)
      as.numeric(forecast::forecast(fit, h=n_val)$mean)
    }, error=function(e) NULL)
    
    if (!is.null(vp)) {
      vmse <- mse(yv, vp)
      if (is.finite(vmse) && vmse < best_ets_val_mse) {
        best_ets_val_mse <- vmse
        best_ets_name <- cand$name
      }
    }
  }
  
  pred_ets_test <- tryCatch({
    cand <- ets_candidates[[which(vapply(ets_candidates, function(x) x$name==best_ets_name, logical(1)))[1]]]
    fit_tr <- forecast::ets(train, damped=cand$damped)
    all_fc <- as.numeric(forecast::forecast(fit_tr, h=n_val + n_test)$mean)
    tail(all_fc, n_test)
  }, error=function(e) rep(NA_real_, n_test))
  
  # -------------------- XGBoost (VAL early stop; recursive forecast from train) --------------------
  pred_xgb_test <- rep(NA_real_, n_test)
  best_xgb_iter <- NA_integer_
  
  tryCatch({
    y_all <- as.numeric(train_val)
    y_tr  <- as.numeric(train)
    
    if ((n_train <= lags + 2) || (n_train + n_val <= lags + 2)) stop("XGB: train/val too short")
    
    d_tr <- make_lag_xy(y_all, lags, t_start = lags + 1, t_end = n_train)
    d_va <- make_lag_xy(y_all, lags, t_start = n_train + 1, t_end = n_train + n_val)
    
    dtrain <- xgb.DMatrix(d_tr$X, label = d_tr$y)
    dval   <- xgb.DMatrix(d_va$X, label = d_va$y)
    
    params <- list(
      objective = "reg:squarederror",
      eta = 0.05,
      max_depth = 3,
      subsample = 0.8,
      colsample_bytree = 0.8,
      eval_metric = "rmse"
    )
    
    model_es <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = xgb_max_rounds,
      watchlist = list(train=dtrain, val=dval),
      early_stopping_rounds = xgb_early_stop,
      verbose = 0
    )
    
    best_xgb_iter <- model_es$best_iteration
    if (is.null(best_xgb_iter) || length(best_xgb_iter) == 0 || !is.finite(best_xgb_iter)) {
      elog <- model_es$evaluation_log
      best_xgb_iter <- if (!is.null(elog) && nrow(elog) > 0) nrow(elog) else xgb_max_rounds
    }
    best_xgb_iter <- max(1L, as.integer(best_xgb_iter))
    
    pred_all <- recursive_forecast_xgb(model_es, history=y_tr, n_ahead=n_val + n_test, lags=lags)
    pred_xgb_test <- tail(pred_all, n_test)
    
  }, error=function(e) {
    message("XGBoost failed: ", e$message)
  })
  
  pred_xgb_test <- align_len(pred_xgb_test, n_test)
  
  # -------------------- LSTM (VAL early stop; recursive forecast from train) --------------------
  pred_lstm_test <- rep(NA_real_, n_test)
  best_lstm_epoch <- NA_integer_
  
  tryCatch({
    clear_keras_session()
    
    train_vec <- as.numeric(train)
    rng <- get_rng(train_vec)
    
    train_val_vec <- as.numeric(train_val)
    train_val_scaled <- scale_with_rng(train_val_vec, rng)
    
    seq_all <- create_sequences_indexed(train_val_scaled, lag = lags)
    if (is.null(seq_all)) stop("LSTM: series too short")
    
    tr_id <- which(seq_all$idx <= n_train)
    va_id <- which(seq_all$idx > n_train & seq_all$idx <= n_train + n_val)
    if (length(tr_id) < 10 || length(va_id) < 5) stop("LSTM: too few train/val samples")
    
    X_tr <- seq_all$X[tr_id,,,drop=FALSE]; y_tr <- seq_all$y[tr_id,,drop=FALSE]
    X_va <- seq_all$X[va_id,,,drop=FALSE]; y_va <- seq_all$y[va_id,,drop=FALSE]
    
    build_lstm <- function() {
      inputs <- layer_input(shape=c(lags,1))
      outputs <- inputs %>% layer_lstm(units=as.integer(lstm_units)) %>% layer_dense(units=1)
      m <- keras_model(inputs, outputs)
      m %>% compile(loss="mse", optimizer="adam")
      m
    }
    
    model_es <- build_lstm()
    cb_es <- callback_early_stopping(
      monitor="val_loss",
      patience=as.integer(lstm_patience),
      restore_best_weights=TRUE
    )
    
    hist <- model_es %>% fit(
      X_tr, y_tr,
      validation_data=list(X_va, y_va),
      epochs=as.integer(lstm_max_epochs),
      batch_size=as.integer(lstm_batch),
      callbacks=list(cb_es),
      verbose=0
    )
    
    vloss <- hist$metrics$val_loss
    if (is.null(vloss)) stop("LSTM: val_loss not found")
    vloss <- as.numeric(unlist(vloss))
    best_lstm_epoch <- which.min(vloss)
    
    train_scaled <- scale_with_rng(train_vec, rng)
    pred_all_scaled <- recursive_forecast_lstm(model_es, train_scaled, n_val + n_test, lags)
    pred_test_scaled <- tail(pred_all_scaled, n_test)
    pred_lstm_test <- inv_scale_with_rng(pred_test_scaled, rng)
    
  }, error=function(e) {
    message("LSTM failed: ", e$message)
  })
  
  pred_lstm_test <- align_len(pred_lstm_test, n_test)
  
  # -------------------- Output (TEST-only) --------------------
  pred_test_df <- data.frame(
    Split  = "TEST",
    Date   = df_test$date,
    Actual = as.numeric(test),
    ARIMA  = align_len(pred_arima_test, n_test),
    TGARCH = align_len(pred_tgarch_test, n_test),
    EGARCH = align_len(pred_egarch_test, n_test),
    ETS    = align_len(pred_ets_test, n_test),
    XGBoost= pred_xgb_test,
    LSTM   = pred_lstm_test,
    LLM    = llm_pred_test
  )
  
  models <- c("ARIMA","TGARCH","EGARCH","ETS","XGBoost","LSTM","LLM")
  metrics <- tibble(
    Model = models,
    MAE   = c(mae(pred_test_df$Actual, pred_test_df$ARIMA),
              mae(pred_test_df$Actual, pred_test_df$TGARCH),
              mae(pred_test_df$Actual, pred_test_df$EGARCH),
              mae(pred_test_df$Actual, pred_test_df$ETS),
              mae(pred_test_df$Actual, pred_test_df$XGBoost),
              mae(pred_test_df$Actual, pred_test_df$LSTM),
              mae(pred_test_df$Actual, pred_test_df$LLM)),
    RMSE  = c(rmse(pred_test_df$Actual, pred_test_df$ARIMA),
              rmse(pred_test_df$Actual, pred_test_df$TGARCH),
              rmse(pred_test_df$Actual, pred_test_df$EGARCH),
              rmse(pred_test_df$Actual, pred_test_df$ETS),
              rmse(pred_test_df$Actual, pred_test_df$XGBoost),
              rmse(pred_test_df$Actual, pred_test_df$LSTM),
              rmse(pred_test_df$Actual, pred_test_df$LLM)),
    MAPE  = c(mape(pred_test_df$Actual, pred_test_df$ARIMA),
              mape(pred_test_df$Actual, pred_test_df$TGARCH),
              mape(pred_test_df$Actual, pred_test_df$EGARCH),
              mape(pred_test_df$Actual, pred_test_df$ETS),
              mape(pred_test_df$Actual, pred_test_df$XGBoost),
              mape(pred_test_df$Actual, pred_test_df$LSTM),
              mape(pred_test_df$Actual, pred_test_df$LLM))
  ) %>% mutate(Split="TEST")
  
  meta <- tibble(
    item = c("dataset","outcome","country","disease","group",
             "split_scheme","n_train","n_val","n_test",
             "val_start","val_end","test_start","test_end",
             "llm_file","llm_name_score","llm_check_mae",
             "best_arima_strategy","best_ets_strategy",
             "tgarch_best","egarch_best",
             "xgb_best_iteration","lstm_best_epoch"),
    value = c(dataset_i, outcome_i, country_i, disease_i, group_i,
              split_scheme, n_train, n_val, n_test,
              as.character(min(df_val$date)), as.character(max(df_val$date)),
              as.character(min(df_test$date)), as.character(max(df_test$date)),
              basename(sel$fp), as.character(sel$ns %||% NA_integer_),
              as.character(sel$check_mae %||% NA_real_),
              best_arima_name %||% NA_character_,
              best_ets_name %||% NA_character_,
              tgarch_best_str, egarch_best_str,
              as.character(best_xgb_iter %||% NA_integer_),
              as.character(best_lstm_epoch %||% NA_integer_))
  )
  
  key_str <- sanitize_filename(paste(dataset_i, outcome_i, country_i, disease_i, group_i, sep="__"))
  out_pred <- file.path(OUTPUT_DIR, paste0(key_str, "_pred_test.xlsx"))
  out_met  <- file.path(OUTPUT_DIR, paste0(key_str, "_metrics_meta_test.xlsx"))
  
  wb <- createWorkbook()
  addWorksheet(wb, "PRED_TEST"); writeData(wb, "PRED_TEST", pred_test_df)
  addWorksheet(wb, "METRICS");   writeData(wb, "METRICS", metrics)
  addWorksheet(wb, "META");      writeData(wb, "META", meta)
  saveWorkbook(wb, out_pred, overwrite=TRUE)
  
  write.xlsx(list(METRICS=metrics, META=meta), out_met, overwrite=TRUE)
  
  message("Done: ", dataset_i, " | ", outcome_i, " | ", country_i, " / ", disease_i, " / ", group_i,
          " | ", split_scheme, " | test=", n_test,
          " | LLM=", basename(sel$fp))
  list(ok=TRUE, key=key_str, n_total=n_total, n_val=n_val, n_test=n_test)
}

# -------------------- 9) Main loop --------------------
run_log <- vector("list", nrow(series_keys))

for (i in seq_len(nrow(series_keys))) {
  dataset_i <- series_keys$dataset[i]
  outcome_i <- series_keys$outcome[i]
  country_i <- series_keys$country[i]
  disease_i <- series_keys$disease[i]
  group_i   <- series_keys$group[i]
  
  run_log[[i]] <- tryCatch({
    run_one_series(dataset_i, outcome_i, country_i, disease_i, group_i)
  }, error=function(e) {
    message("Series failed: ", dataset_i, " | ", outcome_i, " | ", country_i, " / ", disease_i, " / ", group_i, " | ", e$message)
    list(ok=FALSE, reason=e$message, key=paste(dataset_i, outcome_i, country_i, disease_i, group_i, sep=" | "))
  })
}

log_df <- bind_rows(lapply(run_log, function(x) {
  tibble(
    ok      = isTRUE(x$ok),
    key     = x$key %||% NA_character_,
    reason  = x$reason %||% NA_character_,
    n_total = x$n_total %||% NA_integer_,
    n_val   = x$n_val %||% NA_integer_,
    n_test  = x$n_test %||% NA_integer_
  )
}))

log_path <- file.path(OUTPUT_DIR, "run_log.xlsx")
write.xlsx(log_df, log_path, rowNames=FALSE)

message("===============================================")
message("Completed. Success: ", sum(log_df$ok), "  Failed: ", sum(!log_df$ok))
message("Run log saved to: ", log_path)
