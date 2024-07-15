import math
import numpy as np
import pandas as pd


# Convert concentration from mg/L to meq/L
def concentration_meq(concentration_mg, molecular_weight, valence):
    """
    Convert concentration from mg/L to meq/L.

    Parameters:
    concentration_mg (float): Concentration in mg/L.
    molecular_weight (float): Molecular weight of the ion.
    valence (int): Valence of the ion.

    Returns:
    float: Concentration in meq/L.
    """
    equivalent_weight = molecular_weight / valence
    return concentration_mg / equivalent_weight

# Convert relevant ions in a DataFrame from mg/L to meq/L
def convert_ions_to_meq(df, ion_info):
    """
    Convert all relevant ion concentrations in a DataFrame from mg/L to meq/L.

    Parameters:
    df (pd.DataFrame): DataFrame containing ion concentrations in mg/L.
    ion_info (dict): Dictionary containing molecular weights and valences for ions.

    Returns:
    pd.DataFrame: DataFrame with ion concentrations converted to meq/L.
    """
    parameters = {
    'Na': {'molecular_weight': 22.99, 'valence': 1},
    'K': {'molecular_weight': 39.1, 'valence': 1},
    'Ca': {'molecular_weight': 40.08, 'valence': 2},
    'Mg': {'molecular_weight': 24.31, 'valence': 2},
    'Cl': {'molecular_weight': 35.45, 'valence': 1},
    'HCO3': {'molecular_weight': 61.01, 'valence': 1},
    'CO3': {'molecular_weight': 60.01, 'valence': 2},
    'OH': {'molecular_weight': 17.01, 'valence': 1},
    'H': {'molecular_weight': 1.008, 'valence': 1}}

    for ion, parameter in parameters.items():
        if ion in df.columns:
            df[f'{ion}_meq'] = df[ion].apply(lambda x: concentration_meq(x, parameter['molecular_weight'], parameter['valence']))

    return df

# Calculate ionic balabnce
def calculate_ionic_balance(df, cation_cols, anion_cols):
    """
    Calculate the ionic balance in percentage based on The industry standard convention
    (based on APHA Method 1030) 

    Parameters:
    df (pd.DataFrame): DataFrame containing ion concentrations in meq/L.
    cation_cols (list of str): List of column names for cations in meq/L.
    anion_cols (list of str): List of column names for anions in meq/L.

    Returns:
    pd.Series: Ionic balance in percentage for each sample.
    """

    cation_cols = ['Na_meq', 'K_meq', 'Ca_meq', 'Mg_meq']
    anion_cols = ['Cl_meq', 'HCO3_meq', 'CO3_meq', 'OH_meq', 'H_meq']


    cations_sum = df[cation_cols].sum(axis=1)
    anions_sum = df[anion_cols].sum(axis=1)
    ionic_balance = (cations_sum - anions_sum) / (cations_sum + anions_sum) * 100

    if -5 <= ionic_balance <= 5:
        return "Ion is balanced for potable water and environment"
    elif -10 <= ionic_balance <= 10:
        return "Ion is balanced for environment, but not for portable drinking"
    else:
        return "Ion is not balanced, please confirm if there were no errors during sampling or data input"

    return ionic_balance



# Calculate the pH
def calculate_pH(H_concentration):
    """
    Calculate the pH of a solution given the hydrogen ion concentration.
    pH--> calculated by the negative log base 10 of the hydronium ion.
    pH is measured on a scale of 0 to 14. On this scale, acidity ranges from 0 to <7,
    alkalinity from >7 to 14,  a pH value of 7 is neutral (neither acidic nor basic)

    Parameters:
    H_concentration (float): Hydrogen ion concentration in moles/L.

    Returns:
    float: pH value.
    """
    pH = -math.log10(H_concentration)

    return pH

# Solubility Product (Ksp)
def solubility_product(A_+, B_-):
    """
    Calculate the solubility product (Ksp) of a compound.
    Ksp-->  the equilibrium constant for the dissolution of a
    solid substance into an aqueous solution.

    Parameters:
    A_+ (float) : Cation concentration of substance in moles/L.
    B_- (float): Anion concentration of substance in moles/L.

    Returns:
    float: Solubility product.

    e.g Ksp of CaCO3 = Ca * CO3
    """
    Ksp = A_+ * B_-

    return Ksp

# Alkalinity
def calculate_alkalinity(HCO3, CO3, OH, H):
    """
    Calculate the alkalinity of a water sample.

    Acidity --> represents the capacity of a substance to
    donate a proton, or hydrogen ion (H+) 

    Alkalinity -->  is the capacity of a substance to
    accept a proton. It is the chemical opposite of acidity
    and associated with hig pH readings.

    Parameters:
    HCO3 (float): Bicarbonate concentration in meq/L.
    CO3 (float): Carbonate concentration in meq/L.
    OH (float): Hydroxide concentration in meq/L.
    H (float): Hydrogen ion concentration in meq/L.

    Returns:
    float: Alkalinity in meq/L.

    """
    Alkalinity = HCO3 + 2 * CO3 + OH - H #The 2 is the valence of carbonate ion

    return Alkalinity

# Charge Balance
def charge_balance(cations, anions):
    """
    Check if the charge balance of cations and anions is equal.

    Parameters:
    cations (list of float): List of cation concentrations in meq/L.
    anions (list of float): List of anion concentrations in meq/L.

    Returns:
    bool: True if the sum of cations equals the sum of anions, False otherwise.
    """

    charge_balance= sum(cations) == sum(anions)

    return charge_balance

# Ion Activity Product (IAP)
def ion_activity_product(Ksp):
    """
    Calculate the ion activity product (IAP) for a compound.

    Parameters:
    Ca (float): Calcium ion concentration in moles/L.
    CO3 (float): Carbonate ion concentration in moles/L.

    Returns:
    float: Ion activity product.

    IAP/Ksp=1

    """

    IAP = 1/Ksp

    return IAP

# Gibbs Free Energy
def gibbs_free_energy(delta_G0, R, T, Q):
    """
    Calculate the Gibbs free energy change of a reaction.

    Parameters:
    delta_G0 (float): Standard Gibbs free energy change in J/mol.
    R (float): Universal gas constant in J/(mol·K).
    T (float): Temperature in Kelvin.
    Q (float): Reaction quotient.

    Returns:
    float: Gibbs free energy change in J/mol.
    """

    gibbs_free_energy = delta_G0 + R * T * math.log(Q)

    return gibbs_free_energy

# Saturation Index (SI)
def saturation_index(IAP, Ksp):
    """
    Calculate the saturation index (SI) of a mineral.

    Parameters:
    IAP (float): Ion activity product.
    Ksp (float): Solubility product constant.

    Returns:
    float: Saturation index.
    """

    SI = math.log10(IAP / Ksp)

    return SI

def calculate_activity_coefficient(ionic_strength, charge):
    """
    Calculate the activity coefficient using the Debye-Hückel equation.

    Parameters:
    ionic_strength (float): Ionic strength of the solution.
    charge (int): Charge of the ion.

    Returns:
    float: Activity coefficient.
    """
    A = 0.509  # Debye-Hückel constant at 25°C in water
    activity_coefficient = 10**(-A * charge**2 * math.sqrt(ionic_strength) / (1 + math.sqrt(ionic_strength)))

    return activity_coefficient

def ionic_activity(concentration, charge, ionic_strength):
    """
    Calculate the ionic activity.

    Parameters:
    concentration (float): Molar concentration of the ion. (e.g., 0.01 M)
    charge (int): Charge of the ion. (e.g., +1 for Na+)
    ionic_strength (float): Ionic strength of the solution. (e.g., 0.1 M)

    Returns:
    float: Ionic activity.
    """
    Ionic_activity = activity_coefficient * concentration

    return Ionic_activity

# Hardness
def calculate_hardness(Ca, Mg):
    """
    Calculate the hardness of water.

    Parameters:
    Ca (float): Calcium concentration in mg/L.
    Mg (float): Magnesium concentration in mg/L.

    Returns:
    float: Hardness in mg/L.
    """

    Hardness_dH = 2.5 * Ca + 4.1 * Mg

    return Hardness_dH



