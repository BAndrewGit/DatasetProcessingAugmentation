import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


###############################
# Funcții de încărcare și preprocesare
###############################

def load_dataset(filename):
    df = pd.read_csv(filename)
    print("Coloane existente în dataset:")
    print(df.columns.tolist())
    print("\nValori lipsă pe coloane:")
    print(df.isna().sum()[df.isna().sum() > 0])
    return df


def convert_duration(s):
    if isinstance(s, str):
        if 'month' in s:
            return int(s.split()[0])
        elif 'year' in s:
            return int(s.split()[0]) * 12
    return np.nan


def preprocess_data(df):
    savings_goal_cols = ['Savings_Goal_Emergency_Fund', 'Savings_Goal_Major_Purchases',
                         'Savings_Goal_Child_Education', 'Savings_Goal_Vacation',
                         'Savings_Goal_Retirement', 'Savings_Goal_Other']

    savings_obstacle_cols = ['Savings_Obstacle_Insufficient_Income', 'Savings_Obstacle_Other_Expenses',
                             'Savings_Obstacle_Not_Priority', 'Savings_Obstacle_Other']

    expense_dist_cols = ['Expense_Distribution_Food', 'Expense_Distribution_Housing',
                         'Expense_Distribution_Transport', 'Expense_Distribution_Entertainment',
                         'Expense_Distribution_Health', 'Expense_Distribution_Personal_Care',
                         'Expense_Distribution_Child_Education', 'Expense_Distribution_Other']

    credit_cols = ['Credit_Essential_Needs', 'Credit_Major_Purchases',
                   'Credit_Unexpected_Expenses', 'Credit_Personal_Needs', 'Credit_Never_Used']

    passthrough_cols = savings_goal_cols + savings_obstacle_cols + expense_dist_cols + credit_cols

    # Coloanele pentru durate
    duration_cols = ['Product_Lifetime_Clothing', 'Product_Lifetime_Tech',
                     'Product_Lifetime_Appliances', 'Product_Lifetime_Cars']
    for col in duration_cols:
        df[col] = df[col].apply(convert_duration)

    # Seturi de coloane
    numerical_cols = ['Age', 'Income_Category', 'Essential_Needs_Percentage'] + duration_cols
    ordinal_cols = ['Impulse_Buying_Frequency']
    nominal_cols = ['Family_Status', 'Gender', 'Financial_Attitude',
                    'Budget_Planning', 'Save_Money',
                    'Impulse_Buying_Category', 'Impulse_Buying_Reason',
                    'Debt_Level', 'Financial_Investments',
                    'Bank_Account_Analysis_Frequency']

    # Preprocesare pentru coloanele numerice
    for col in numerical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Preprocesare pentru coloanele categorice
    for col in ordinal_cols + nominal_cols:
        if col == 'Debt_Level':
            # Dacă lipsesc valori, le înlocuim cu "Absent"
            df[col] = df[col].fillna("Absent")
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        # Convertim în string
        df[col] = df[col].astype(str)

    # Definim preprocesorul cu transformări separate:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('ord', OrdinalEncoder(categories=[['Very rarely', 'Rarely', 'Sometimes', 'Often', 'Very often']]), ordinal_cols),
        ('nom', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), nominal_cols),
        ('pass', 'passthrough', passthrough_cols)
    ])

    return preprocessor, numerical_cols, ordinal_cols, nominal_cols, passthrough_cols


###############################
# Funcții pentru SMOTE, VAE și augmentare
###############################

def apply_smote(X, y):
    if np.isnan(X).any():
        print("Atenție: S-au găsit NaN în datele preprocesate! Se înlocuiesc cu 0.")
        X = np.nan_to_num(X)
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


# Clasa VAE este definită, însă apelul său este comentat pentru moment
class VAE:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.encoder, self.decoder, self.vae = self._build_model()

    def _build_model(self):
        # Encoder
        inputs = Input(shape=(self.input_dim,))
        h = Dense(64, activation='relu')(inputs)
        z_mean = Dense(2)(h)
        z_log_var = Dense(2)(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling)([z_mean, z_log_var])

        # Decodor
        decoder_h = Dense(64, activation='relu')
        decoder_out = Dense(self.input_dim)
        h_decoded = decoder_h(z)
        outputs = decoder_out(h_decoded)

        vae = Model(inputs, outputs)
        reconstruction_loss = K.mean(K.square(inputs - outputs))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae.add_loss(K.mean(reconstruction_loss + kl_loss))
        vae.compile(optimizer=Adam(0.001))

        encoder = Model(inputs, z_mean)
        latent_inputs = Input(shape=(2,))
        _h_decoded = decoder_h(latent_inputs)
        _decoded = decoder_out(_h_decoded)
        decoder = Model(latent_inputs, _decoded)

        return encoder, decoder, vae

    def generate(self, n_samples):
        z = np.random.normal(size=(n_samples, 2))
        return self.decoder.predict(z)


def augment_class(X_class, n_samples):
    vae = VAE(X_class.shape[1])
    vae.vae.fit(X_class, epochs=100, batch_size=32, verbose=0)
    return vae.generate(n_samples)


###############################
# Funcții pentru inverse transform (la formatul encoded final)
###############################

def full_inverse_transform(preprocessor, X_final, numerical_cols, ordinal_cols, nominal_cols, passthrough_cols):
    # Inversează transformările pentru fiecare segment
    n_num = len(numerical_cols)
    n_ord = len(ordinal_cols)
    n_nom = preprocessor.named_transformers_['nom'].get_feature_names_out(nominal_cols).shape[0]
    n_pass = len(passthrough_cols)

    # Extrage segmentele
    X_num = X_final[:, :n_num]
    X_ord = X_final[:, n_num:n_num + n_ord]
    X_nom = X_final[:, n_num + n_ord:n_num + n_ord + n_nom]
    X_pass = X_final[:, n_num + n_ord + n_nom:]  # Passthrough este ultimul segment

    # Inversează transformările
    X_num_inv = preprocessor.named_transformers_['num'].inverse_transform(X_num)
    X_ord_inv = preprocessor.named_transformers_['ord'].inverse_transform(X_ord)
    X_nom_inv = preprocessor.named_transformers_['nom'].inverse_transform(X_nom)  # Returnează coloanele nominale originale (string)

    # Combină toate segmentele
    return np.hstack([X_num_inv, X_ord_inv, X_nom_inv, X_pass])


###############################
# Funcția main – fluxul complet
###############################

def main():
    df = load_dataset('DatasetOriginal.csv')
    preprocessor, numerical_cols, ordinal_cols, nominal_cols, passthrough_cols = preprocess_data(df)

    # Separați X de etichetă y
    X = df.drop('Behavior_Risk_Level', axis=1)
    y = df['Behavior_Risk_Level'].values

    # Aplicăm transformările
    X_processed = preprocessor.fit_transform(X)
    X_res, y_res = apply_smote(X_processed, y)

    # Full inverse transform
    X_inversed = full_inverse_transform(preprocessor, X_res, numerical_cols, ordinal_cols, nominal_cols, passthrough_cols)

    # Obține numele coloanelor finale (folosind numele nominale originale, nu one-hot)
    final_columns = numerical_cols + ordinal_cols + nominal_cols + passthrough_cols

    # Verificare finală a numărului de coloane
    if len(final_columns) != X_inversed.shape[1]:
        raise ValueError(f"Așteptat: {len(final_columns)} coloane, dar am primit {X_inversed.shape[1]}.")

    df_final = pd.DataFrame(data=X_inversed, columns=final_columns)

    # Post-procesare:
    # 1. Pentru "Age": rotunjim la cel mai apropiat număr întreg.
    df_final['Age'] = df_final['Age'].round(0).astype(int)

    # 2. Pentru "Income_Category": rotunjim la cel mai apropiat multiplu de 100.
    df_final['Income_Category'] = (df_final['Income_Category'] / 100).round(0) * 100
    df_final['Income_Category'] = df_final['Income_Category'].astype(int)

    # 3. Pentru "Essential_Needs_Percentage": rotunjim la cel mai apropiat multiplu de 5.
    df_final['Essential_Needs_Percentage'] = (df_final['Essential_Needs_Percentage'] / 5).round(0) * 5
    df_final['Essential_Needs_Percentage'] = df_final['Essential_Needs_Percentage'].astype(int)

    # 4. Rotunjim coloanele binare (passthrough) la 0 sau 1
    for col in passthrough_cols:
        df_final[col] = df_final[col].round().astype(int)

    # 5. Pentru coloanele de durată, convertim valorile numerice în formatul inițial.
    for col in ['Product_Lifetime_Clothing', 'Product_Lifetime_Tech', 'Product_Lifetime_Appliances',
                'Product_Lifetime_Cars']:
        df_final[col] = df_final[col].apply(lambda x: f"{int(x)} months" if x < 12 else f"{int(x // 12)} years")

    # Adăugăm etichetele
    df_final['Behavior_Risk_Level'] = y_res
    df_final['Risk_Label'] = df_final['Behavior_Risk_Level'].map({0: 'Beneficially', 1: 'Risky'})

    # Salvăm setul de date final
    df_final.to_csv('dataset_augmentatVAE+SMOTE.csv', index=False)
    print("Datasetul a fost salvat cu succes!")


if __name__ == "__main__":
    main()
