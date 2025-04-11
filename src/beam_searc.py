import numpy as np

def beam_search(prob_matrix, padding_id, beam_size):
    """
    Алгоритм beam search для декодера
    
    Параметры:
    prob_matrix (np.ndarray): Матрица вероятностей формы (batch_size, src_len, num_tokens)
    padding_id (int): ID токена паддинга, который нужно исключить из рассмотрения
    beam_size (int): Размер луча (количество сохраняемых гипотез)
    
    Возвращает:
    list: Список последовательностей токенов для каждого элемента в батче
    """
    batch_size, src_len, num_tokens = prob_matrix.shape
    all_results = []
    
    # Обрабатываем каждый пример в батче независимо
    for batch_idx in range(batch_size):
        # Инициализируем начальные гипотезы (лог-вероятность, последовательность)
        hypotheses = [(0.0, [])]
        
        # Проходим по всем временным шагам
        for step in range(src_len):
            candidates = []
            current_probs = prob_matrix[batch_idx, step, :].copy()
            
            # Заменяем вероятность паддинга на -inf, чтобы исключить его из рассмотрения
            current_probs[padding_id] = -np.inf
            
            # Обрабатываем каждую гипотезу
            for log_prob, sequence in hypotheses:
                # Получаем индексы топ-K токенов для текущего шага
                top_k_indices = np.argsort(current_probs)[::-1][:beam_size * 2]
                
                # Расширяем гипотезы
                for token_idx in top_k_indices:
                    # Пропускаем недопустимые токены (на случай если padding попал в топ)
                    if token_idx == padding_id or current_probs[token_idx] <= 0:
                        continue
                    
                    # Вычисляем новую логарифмическую вероятность
                    new_log_prob = log_prob + np.log(current_probs[token_idx])
                    
                    # Создаем новую последовательность
                    new_sequence = sequence + [token_idx]
                    candidates.append((new_log_prob, new_sequence))
            
            # Сортируем кандидатов по вероятности и выбираем топ-N
            candidates.sort(key=lambda x: x[0], reverse=True)
            hypotheses = candidates[:beam_size]
        
        # Выбираем гипотезу с максимальной вероятностью
        best_hypothesis = max(hypotheses, key=lambda x: x[0])[1]
        all_results.append(best_hypothesis)
    
    return all_results