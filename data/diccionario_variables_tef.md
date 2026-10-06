# Diccionario de variables del dataset sintético de TEF

Transferencias Electrónicas de Fondos. Montos en CLP. La columna objetivo del proyecto es `is_fraud`.

| Variable | Tipo | Descripción |
|---|---|---|
| `transaction_id` | Texto | Identificador único de la transacción. |
| `timestamp` | Fecha | Fecha y hora de la transacción. |
| `sender_id` | Texto | Identificador del emisor. |
| `receiver_id` | Texto | Identificador del receptor. Puede corresponder a un usuario, un comercio o una cuenta asociada a mula. |
| `amount` | Numérico | Monto en CLP. |
| `currency` | Texto | Moneda. En este archivo es CLP. |
| `transaction_type` | Texto | `TRANSFERENCIA`, `PAGO_COMERCIO`, `PAGO_SERVICIO` o `PAGO_CUENTA`. |
| `sender_bank` | Texto | Banco del emisor. |
| `sender_balance_before` | Numérico | Saldo del emisor antes de la transacción. |
| `sender_balance_after` | Numérico | Saldo del emisor después de la transacción. |
| `receiver_balance_before` | Numérico | Saldo del receptor antes. En comercios vale 0. |
| `receiver_balance_after` | Numérico | Saldo del receptor después. En comercios vale 0. |
| `is_fraud` | Entero | 1 si la transacción es fraude y 0 si es legítima. |
| `fraud_type` | Texto | Tipo de fraude cuando `is_fraud` es 1. Vale `none` en transacciones legítimas. |
| `sender_location_lat` | Numérico | Latitud de origen del emisor. |
| `sender_location_lon` | Numérico | Longitud de origen del emisor. |
| `receiver_location_lat` | Numérico | Latitud del receptor. |
| `receiver_location_lon` | Numérico | Longitud del receptor. |
| `distance_from_home` | Numérico | Distancia en kilómetros entre el origen y el hogar registrado del emisor. |
| `time_since_last_transaction` | Numérico | Horas desde la transacción anterior del emisor. |
| `sender_device_id` | Texto | Dispositivo del emisor. |
| `receiver_device_id` | Texto | Dispositivo del receptor. Puede ser `N/A` o un terminal de comercio. |
| `sender_ip_address` | Texto | IP de origen del emisor. |
| `receiver_ip_address` | Texto | IP del receptor. |
| `sender_account_age` | Entero | Antigüedad de la cuenta del emisor, en días. |
| `sender_avg_daily_transactions` | Numérico | Promedio diario de transacciones del emisor. |
| `sender_avg_daily_amount` | Numérico | Promedio del monto diario del emisor. |
| `sender_recent_account_changes` | Texto | Cambios recientes en la cuenta. Vale `none` si no hay cambios. |
| `sender_communication_channel_flag` | Texto | Indicador de comunicación asociada al emisor. Vale `none` si no hay alerta. |
| `last_24h_count` | Entero | Transacciones del emisor en las últimas 24 horas. |
| `avg_amount_last_7d` | Numérico | Monto promedio de los últimos 7 días. Es la columna con nulos. |
| `amount_vs_avg` | Numérico | Diferencia del monto respecto del promedio diario del emisor. |
| `recent_tx_ratio` | Numérico | Relación entre la actividad reciente y el promedio diario. |
| `sender_balance_delta` | Numérico | Cambio de saldo del emisor. En este archivo suele coincidir con `amount`. |
| `receiver_balance_delta` | Numérico | Cambio de saldo del receptor. Vale 0 cuando el receptor es un comercio. |
| `distance_sender_receiver` | Numérico | Distancia entre emisor y receptor. |

Campos que calcula la aplicación y no vienen en el CSV: `hora`, `clase`, `alerta_comunicacion` y `cambio_cuenta`.
