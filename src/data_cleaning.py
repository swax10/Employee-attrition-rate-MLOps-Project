import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
#here, abc refers to the module and ABC refers to the Abstract Base Classes
from typing import Union
# Import OneHotEncoder from sklearn
from sklearn.preprocessing import OneHotEncoder

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass
class DataPreProcessStrategy(DataStrategy):
    def __init__(self, encoder=None):
        self.encoder = encoder
    """This class is used to preprocess the given dataset"""
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Column Names Before Preprocessing:", data.columns)  # Add this line
            data = data.drop(["EmployeeCount", "EmployeeNumber", "StandardHours"], axis=1)
            if 'Attrition' in data.columns:
                print("Attrition column found in data.")
            else:
                print("Attrition column not found in data.")
            data["Attrition"] = data["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
            data["Over18"] = data["Over18"].apply(lambda x: 1 if x == "Yes" else 0)
            data["OverTime"] = data["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)

            # Extract categorical variables
            cat = data[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]

            # Perform one-hot encoding on categorical variables
            onehot = OneHotEncoder()
            cat_encoded = onehot.fit_transform(cat).toarray()
            #to have the feature unqiue data as their respective encoded column names
            feature_names = onehot.get_feature_names_out(input_features=cat.columns)
            # Convert cat_encoded to DataFrame
            cat_df = pd.DataFrame(cat_encoded,columns=feature_names)
            print(cat_df.head())
            # Extract numerical variables
            numerical = data[['Age', 'Attrition', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]

            # Concatenate X_cat_df and X_numerical
            data = pd.concat([cat_df, numerical], axis=1)

            print("Column Names After Preprocessing:", data.columns)  # Add this line
            print("Preprocessed Data:")
            print(data.head())
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e



class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # Check if 'Attrition' is present in the data
            if 'Attrition' in data.columns:
                X = data.drop(['Attrition'], axis=1)
                Y = data['Attrition']
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                return X_train, X_test, Y_train, Y_test
            else:
                raise ValueError("'Attrition' column not found in data.")
        except Exception as e:
            logging.error(f"Error in data handling: {str(e)}")
            raise e

class DataCleaning:
            def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
                self.data=data
                self.strategy=strategy
            def handle_data(self)->Union[pd.DataFrame,pd.Series]:
                try:
                    return self.strategy.handle_data(self.data)
                except Exception as e:
                    logging.error(f"There is a error in dataHandling{e}")
                    raise e
                        
                 
"""A very nice explanantion of the Strategy design pattern
The Strategy Design Pattern is a simple and powerful concept in software design. It's used when you want to define a family of interchangeable algorithms or behaviors and make them easily switchable without affecting the client code.

Imagine you're a medieval warrior, and you have different weapons to choose from: a sword, a bow, and a axe. Depending on the situation, you might want to use a different weapon. The Strategy Design Pattern helps you model this scenario:

Context: This is like you, the warrior. It's the entity that will use the strategies. It has a reference to a strategy, but it doesn't need to know the specific details of each strategy.

Strategy: These are the different weapons: sword, bow, axe. Each weapon is a separate class that implements the same interface. This interface defines a common method that all weapons will have, like attack().

Concrete Strategies: These are the actual instances of the weapons (sword instance, bow instance, etc.). They implement the methods defined in the strategy interface.

Here's how it works in code:

python
Copy code
# Define the Strategy interface
class WeaponStrategy:
    def attack(self):
        pass
Red(!)
# Concrete Strategy classes
class Sword(WeaponStrategy):
    def attack(self):
        print("Attacking with a sword!")

class Bow(WeaponStrategy):
    def attack(self):
        print("Shooting with a bow!")

# Context class
class Warrior:
    def __init__(self, weapon):
        self.weapon = weapon

    def attack_with_weapon(self):
        self.weapon.attack()

# Create different weapon instances
sword = Sword()
bow = Bow()

# Create a warrior with different weapons
warrior_with_sword = Warrior(sword)
warrior_with_bow = Warrior(bow)

# Warriors attack using their respective weapons
warrior_with_sword.attack_with_weapon()
warrior_with_bow.attack_with_weapon()
In this example, the Warrior class is the context, WeaponStrategy is the strategy interface, and Sword and Bow are the concrete strategy classes. Depending on the weapon you give the warrior, their attack behavior changes.

The beauty of the Strategy Design Pattern is that it promotes separation of concerns and flexibility. You can add new weapons (strategies) without modifying the existing code (context), and you can easily switch between weapons at runtime. This pattern is widely used in software design to manage different algorithms, data handling methods, and more, without tightly coupling them to the main codebase."""
                

              