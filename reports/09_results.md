# Комплексні результати експериментів та аналіз

У цьому звіті представлені консолідовані результати проекту **Transition Matrix ETM-XAI**. Звіт деталізує кількісні метрики, візуалізує ефекти еквіваріантної матриці переходу та надає критичний науковий аналіз отриманих результатів.

## 1. Експерименти з синтетичними даними

Синтетичні експерименти підтверджують основну теоретичну базу, використовуючи низькорозмірні дані ($m=15, k=5, l=4$).

### 1.1 Візуалізація та Структура Матриць

<img src="../outputs/synthetic/figures/png/01_mds_A.png" width="80%" />

*Рис. 1: MDS візуалізація матриці $A$ (Латентний простір). Чітко видно кластеризацію трьох класів.*

<img src="../outputs/synthetic/figures/png/02_mds_B.png" width="80%" />

*Рис. 2: MDS візуалізація матриці $B$ (Простір спостережень). Структура многовиду збережена після трансформації.*

<img src="../outputs/synthetic/figures/png/03_heatmap_T_old.png" width="80%" />

*Рис. 3: Теплова карта матриці $T_{old}$ (Базовий метод). Ваги розподілені хаотично, відсутня чітка структура.*

<img src="../outputs/synthetic/figures/png/04_heatmap_T_new.png" width="80%" />

*Рис. 4: Теплова карта матриці $T_{new}$ (Еквіваріантний метод). Помітна блочна структура, що відображає врахування симетрії.*

<img src="../outputs/synthetic/figures/png/05_heatmap_JA.png" width="80%" />

*Рис. 5: Теплова карта генератора $J^A$. Кососиметрична матриця, що відповідає генератору групи обертань SO(2).*

<img src="../outputs/synthetic/figures/png/06_heatmap_JB.png" width="80%" />

*Рис. 6: Теплова карта генератора $J^B$. Емпірично оцінений генератор у просторі спостережень.*

<img src="../outputs/synthetic/figures/png/07_singular_values_M.png" width="80%" />

*Рис. 7: Сингулярні значення системної матриці $M$. Швидке спадання свідчить про хорошу обумовленість регуляризованої задачі.*

### 1.2 Оптимізація та Компроміс (Trade-off)

Аналіз впливу параметра регуляризації $\lambda$.

<img src="../outputs/synthetic/figures/png/08_tradeoff_mse_vs_lambda.png" width="80%" />

*Рис. 8: Залежність MSE реконструкції від $\lambda$. Похибка незначно зростає, залишаючись у допустимих межах.*

<img src="../outputs/synthetic/figures/png/09_tradeoff_sym_vs_lambda.png" width="80%" />

*Рис. 9: Залежність похибки симетрії від $\lambda$. Експоненційне падіння похибки, що підтверджує ефективність методу.*

**Таблиця 1: Кількісні показники компромісу**

| $\lambda$ | MSE (Reconstruction) | Похибка симетрії ($\|TJ^A - J^BT\|_F^2$) | Покращення симетрії |
| :--- | :--- | :--- | :--- |
| **0.0 (Базовий)** | **0.00367** | **13077.17** | **-** |
| 0.1 | 0.00521 | 0.129 | > 99.99% |
| 0.25 | 0.00524 | 0.046 | > 99.99% |
| **0.50 (Обраний)** | **0.00524** | **0.042** | **> 99.99%** |
| 1.0 | 0.00525 | 0.042 | > 99.99% |
| 2.0 | 0.00532 | 0.040 | > 99.99% |

### 1.3 Стійкість до обертання (Robustness)

Візуалізація того, як модель справляється з обертанням вхідних даних.

<img src="../outputs/synthetic/figures/png/10a_robustness_pca.png" width="80%" />

*Рис. 10: Стійкість у проекції PCA (Old vs New). $T_{new}$ краще зберігає структуру класів при обертанні.*

<img src="../outputs/synthetic/figures/png/10b_robustness_mds.png" width="80%" />

*Рис. 11: Стійкість у проекції MDS. Точки $T_{new}$ формують більш компактні кластери.*

<img src="../outputs/synthetic/figures/png/10c_robustness_tsne.png" width="80%" />

*Рис. 12: Стійкість у проекції t-SNE. Чітке розділення класів для методу з симетрією.*

<img src="../outputs/synthetic/figures/png/10d_robustness_umap.png" width="80%" />

*Рис. 13: Стійкість у проекції UMAP. Глобальна структура многовиду збережена значно краще у $T_{new}$.*

<img src="../outputs/synthetic/figures/png/11_displacement_vectors.png" width="80%" />

*Рис. 14: Векторне поле зміщень. Зліва ($T_{old}$): великі відхилення від ідеальної траєкторії. Справа ($T_{new}$): майже нульові зміщення, ідеальна еквіваріантність.*

<img src="../outputs/synthetic/figures/png/12_error_vs_angle.png" width="80%" />

*Рис. 15: Залежність помилки від кута обертання [-90, 90]. $T_{new}$ демонструє стабільність, тоді як помилка $T_{old}$ зростає.*

### 1.4 Порівняння: Статика проти Обертання (Extended)

Накладання оригінальних (статичних) даних на обернені для перевірки інваріантності представлення.

<img src="../outputs/synthetic/figures/png/13a_robustness_pca.png" width="80%" />

*Рис. 16: PCA (Статика vs Обертання). $T_{new}$ показує ідеальне накладання, що свідчить про інваріантність.*

<img src="../outputs/synthetic/figures/png/13b_robustness_mds.png" width="80%" />

*Рис. 17: MDS (Статика vs Обертання). Хаотичність у $T_{old}$ проти впорядкованості у $T_{new}$.*

<img src="../outputs/synthetic/figures/png/13c_robustness_tsne.png" width="80%" />

*Рис. 18: t-SNE (Статика vs Обертання). Підтвердження кластерної стабільності.*

<img src="../outputs/synthetic/figures/png/13d_robustness_umap.png" width="80%" />

*Рис. 19: UMAP (Статика vs Обертання). Топологія многовиду залишається незмінною.*

---

## 2. Експерименти з MNIST

Масштабування методології на реальні зображення.

### 2.1 Реконструкція та Метрики

<img src="../outputs/mnist/figures/png/03_reconstructions_T_old.png" width="80%" />

*Рис. 20: Приклади реконструкції цифр методом $T_{old}$.*

<img src="../outputs/mnist/figures/png/04_reconstructions_T_new.png" width="80%" />

*Рис. 21: Приклади реконструкції цифр методом $T_{new}$. Візуально якість збережена на рівні з базовим методом.*

<img src="../outputs/mnist/figures/png/05_ssim_comparison.png" width="80%" />

*Рис. 22: Гістограма розподілу SSIM (Test set). Розподіли майже ідентичні.*

<img src="../outputs/mnist/figures/png/06_psnr_comparison.png" width="80%" />

*Рис. 23: Гістограма розподілу PSNR (Test set).*

**Таблиця 2: Середні метрики якості (MNIST)**

| Метрика | $T_{old}$ (Базовий) | $T_{new}$ (Еквіваріантний) | Delta |
| :--- | :--- | :--- | :--- |
| **SSIM** | 0.6978 | 0.6976 | -0.03% |
| **PSNR** | 18.49 dB | 18.48 dB | -0.05% |

### 2.2 Аналіз Симетрії

<img src="../outputs/mnist/figures/png/07b_symmetry_bar_train.png" width="80%" />

*Рис. 24: Похибка симетрії (Train metrics). Значне зменшення комутаційної помилки для $T_{new}$.*

<img src="../outputs/mnist/figures/png/07b_symmetry_bar_test.png" width="80%" />

*Рис. 25: Похибка симетрії (Test metrics). Узагальнення властивості симетрії на тестові дані.*

### 2.3 Стійкість до обертання (Curves)

Аналіз залежності якості реконструкції від кута повороту в діапазоні [-90, 90].

<img src="../outputs/mnist/figures/png/08_robustness_ssim_vs_angle_train.png" width="80%" />

*Рис. 26: SSIM vs Кут (Train). Нижній графік показує відношення $SSIM_{new} / SSIM_{old}$.*

<img src="../outputs/mnist/figures/png/08_robustness_ssim_vs_angle_test.png" width="80%" />

*Рис. 27: SSIM vs Кут (Test). Перевага $T_{new}$ помітна при певних кутах (Ratio > 1.0).*

<img src="../outputs/mnist/figures/png/09_robustness_psnr_vs_angle_train.png" width="80%" />

*Рис. 28: PSNR vs Кут (Train).*

<img src="../outputs/mnist/figures/png/09_robustness_psnr_vs_angle_test.png" width="80%" />

*Рис. 29: PSNR vs Кут (Test).*

### 2.4 Вкладення Простору (Embeddings - Simple)

Візуалізація латентного простору $B^*$ (Test Set).

<img src="../outputs/mnist/figures/png/09a_scatter_pca_train.png" width="80%" />

*Рис. 30: PCA Проекція (Train).*

<img src="../outputs/mnist/figures/png/09a_scatter_pca_test.png" width="80%" />

*Рис. 31: PCA Проекція (Test). Порівняння просторів $B^*_{old}$ та $B^*_{new}$.*

<img src="../outputs/mnist/figures/png/09b_scatter_mds_train.png" width="80%" />

*Рис. 32: MDS Проекція (Train).*

<img src="../outputs/mnist/figures/png/09b_scatter_mds_test.png" width="80%" />

*Рис. 33: MDS Проекція (Test).*

<img src="../outputs/mnist/figures/png/09c_scatter_tsne_train.png" width="80%" />

*Рис. 34: t-SNE Проекція (Train).*

<img src="../outputs/mnist/figures/png/09c_scatter_tsne_test.png" width="80%" />

*Рис. 35: t-SNE Проекція (Test).*

<img src="../outputs/mnist/figures/png/09d_scatter_umap_train.png" width="80%" />

*Рис. 36: UMAP Проекція (Train).*

<img src="../outputs/mnist/figures/png/09d_scatter_umap_test.png" width="80%" />

*Рис. 37: UMAP Проекція (Test).*

### 2.5 Візуалізація "Хаосу" (Visual Reconstruction)

Детальний аналіз реконструкції при фіксованих кутах обертання. Рядок "Advantage" (синій/червоний) показує різницю помилок: синій колір означає перевагу $T_{new}$ (покращення), червоний — погіршення.

<img src="../outputs/mnist/figures/png/10a_chaos_figure_test.png" width="80%" />

*Рис. 38: Реконструкція при 0° (Статика). Обидва методи працюють майже ідентично.*

<img src="../outputs/mnist/figures/png/10b_chaos_figure_test.png" width="80%" />

*Рис. 39: Реконструкція при 45°. Помітна деградація якості, але $T_{new}$ намагається краще зберегти структуру.*

<img src="../outputs/mnist/figures/png/10c_chaos_figure_test.png" width="80%" />

*Рис. 40: Реконструкція при 90°. Екстремальні умови. Карта різниці показує зони, де $T_{new}$ виграє.*

### 2.6 Розширені Вкладення (Extended Robustness Scatter)

Накладання "Статичних" (яскравих) та "Обернених" (блідих) точок для оцінки топологічної стабільності.

<img src="../outputs/mnist/figures/png/11a_robustness_pca_train.png" width="80%" />

*Рис. 41: PCA розширена візуалізація (Train).*

<img src="../outputs/mnist/figures/png/11a_robustness_pca_test.png" width="80%" />

*Рис. 42: PCA розширена візуалізація (Test). Перевірка змішування класів при обертанні.*

<img src="../outputs/mnist/figures/png/11b_robustness_mds_train.png" width="80%" />

*Рис. 43: MDS розширена візуалізація (Train).*

<img src="../outputs/mnist/figures/png/11b_robustness_mds_test.png" width="80%" />

*Рис. 44: MDS розширена візуалізація (Test).*

<img src="../outputs/mnist/figures/png/11c_robustness_tsne_train.png" width="80%" />

*Рис. 45: t-SNE розширена візуалізація (Train).*

<img src="../outputs/mnist/figures/png/11c_robustness_tsne_test.png" width="80%" />

*Рис. 46: t-SNE розширена візуалізація (Test).*

<img src="../outputs/mnist/figures/png/11d_robustness_umap_train.png" width="80%" />

*Рис. 47: UMAP розширена візуалізація (Train).*

<img src="../outputs/mnist/figures/png/11d_robustness_umap_test.png" width="80%" />

*Рис. 48: UMAP розширена візуалізація (Test). UMAP найкраще демонструє глобальну структуру многовиду.*

## Висновки

Впровадження Еквіваріантних Матриць Переходу демонструє значний успіх на синтетичних даних, досягаючи майже ідеальної симетрії та стійкості до обертань. На складних даних MNIST метод також показує зменшення похибки симетрії та певні переваги у стійкості (особливо помітні на картах різниці), хоча ефект обмежується нелінійністю реальних зображень. Оновлена візуалізація дозволяє глибше зрозуміти природу цих перетворень.
