import numpy as np

MAX_STEPS_PER_ITERATION = 60

def q_learning(mdp_model, env_step_func, num_actions,max_iterations=100000,
                   alpha=0.1, gamma=0.95, init_epsilon=1.0, epsilon_min=0.01,
                   decay=0.9995, threshold=1e-4, consecutive_iterations=10):

    print("Starting Q-Learning Algorithm...")

    actions_list = list(range(num_actions))
    # אתחול טבלת ה-Q עם אפסים לכל המצבים הידועים
    q_table = {s: {a: 0.0 for a in actions_list} for s in mdp_model.states}

    epsilon = init_epsilon
    converged_count = 0
    convergence_history = []

    for iteration in range(max_iterations):
        # (בחירת מצב התחלה)
        possible_starts = [s for s in mdp_model.states if s[0] < 250]
        possible_starts = [s for s in possible_starts if s in mdp_model.transitions]

        if not possible_starts:
            possible_starts = list(mdp_model.transitions.keys())

        current_state = possible_starts[np.random.choice(len(possible_starts))]
        max_change_this_iteration = 0.0

        # דעיכת קצב הלמידה (alpha)
        current_alpha = max(0.01, alpha * (0.9999 ** iteration))

        #  לולאת הצעדים בתוך איטרציה, כל ניסוי נמשך בין 3 ל-4 שבועות
        for _ in range(MAX_STEPS_PER_ITERATION):
            #(Epsilon-Greedy)
            if np.random.random() < epsilon:
                action = np.random.choice(actions_list)
            else:
                action = max(q_table[current_state], key=q_table[current_state].get)

            # ביצוע הפעולה מול הסביבה
            next_state, reward = env_step_func(current_state, action)

            #  יישום משוואת בלמן
            max_future_q = max(q_table[next_state].values()) if next_state in q_table else 0
            current_q = q_table[current_state][action]

            #ממש העדכון על פי הנוסחה
            new_q = current_q + current_alpha * (reward + gamma * max_future_q - current_q)

            # מעקב אחר ההתכנסות (השינוי המקסימלי)
            change = abs(new_q - current_q)
            if change > max_change_this_iteration:
                max_change_this_iteration = change

            # עדכון הטבלה והתקדמות
            q_table[current_state][action] = new_q
            current_state = next_state

        # סיום איטרציה: עדכון משתנים ובדיקת התכנסות
        convergence_history.append(max_change_this_iteration)

        if epsilon > epsilon_min:
            epsilon *= decay

        if max_change_this_iteration < threshold:
            converged_count += 1
        else:
            converged_count = 0

        if converged_count >= consecutive_iterations:
            print(f"Algorithm converged successfully at iteration {iteration}!")
            break

        if iteration % 5000 == 0:
            print(
                f"Iteration {iteration}/{max_iterations} | Epsilon: {iteration:.4f} | Max Change: {max_change_this_iteration:.6f}")

    print("Q-Learning Training Finished.")
    return q_table, convergence_history