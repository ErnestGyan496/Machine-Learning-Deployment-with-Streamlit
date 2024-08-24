import streamlit as st
import pickle
import numpy as np


def load_model():
    with open("RF_model.pkl", "rb") as file:
        data = pickle.load(file)
    return data


data = load_model()


RF_model = data["model"]
le_tube_geometry = data["le_tube_geometry"]
le_working_fluid = data["le_working_fluid"]
scaler = data["scaler"]


def show_predict_page():
    st.title("Heat Transfer Coefficient Prediction")
    st.image(
        "The_Experimental_facility.png",
        caption="Experimental Facility from NTNU ",
        output_format="auto",
    )

    st.image(
        "Experimental_Schematics.png",
        caption="My Flow Boiling Schematic",
        output_format="auto",
    )

    st.image(
        "Test Section .png",
        caption="Test Section",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )

    st.write(
        """#### we need some information to predict the heat transfer coefficient"""
    )

    Working_Fluid = ("R245fa", "R1233zd(E)", "R1234ZE(E)")

    Tube_Geometry = ("H", "V")

    Working_Fluid = st.selectbox("Working_FLuid", Working_Fluid)

    Tube_Geometry = st.selectbox("Tube_Geometry", Tube_Geometry)

    Internal_Diameter = st.number_input("Internal Diameter [2-21 mm]")
    st.write(Internal_Diameter)

    Saturation_Temperature = st.number_input("Saturation_Temperature [50 - 130 oC]")
    st.write(Saturation_Temperature)

    Saturation_Pressure = st.number_input("Saturation_Pressure [400.0 - 789.01 kPa]")
    st.write(Saturation_Pressure)

    Mass_Velocity = st.number_input("Mass Velocity [200.0 - 1500 kg/m2s]")
    st.write(Mass_Velocity)

    Heat_Flux = st.number_input("Heat_Flux [430 - 28000 kW/m2]")
    st.write(Heat_Flux)

    Vapor_Quality = st.number_input("Vapor_Quality [0 - 1]")
    st.write(Vapor_Quality)

    Reduced_Temperature = st.number_input("Reduced_Temperature [0.0 - 1.0]")
    st.write(Reduced_Temperature)

    Reduced_Pressure = st.number_input("Reduced_Pressure [0.0 - 1.0]")
    st.write(Reduced_Pressure)

    Liquid_Density = st.number_input("Liquid_Density [820.0 - 1300.0 kg/m3]")
    st.write(Liquid_Density)

    Vapor_Density = st.number_input("Vapor_Density [22.00 - 190.00 kg/m3]")
    st.write(Vapor_Density)

    Liquid_Viscosity = st.number_input("Liquid_Viscosity [820.0 - 1252.40 Pa-s]")
    st.write(Liquid_Viscosity)

    Vapor_Viscosity = st.number_input("Vapor_Viscosity [0.000011 - 0.000018 Pa-s]")
    st.write(Vapor_Viscosity)

    Liquid_Thermal_Conductivity = st.number_input(
        "Liquid_Thermal_Conductivity [0.0100 - 0.080 W/mK]"
    )
    st.write(Liquid_Thermal_Conductivity)

    Vapor_Thermal_Conductivity = st.number_input(
        "Vapor_Thermal_Conductivity [0.010 - 0.030 W/mK]"
    )
    st.write(Vapor_Thermal_Conductivity)

    Surface_Tension = st.number_input("Surface_Tension [0.00090 - 0.00990 N/m]")
    st.write(Surface_Tension)

    Liquid_Enthalpy = st.number_input("Liquid_Enthalpy [270 - 400.0 kJ/kg]")
    st.write(Liquid_Enthalpy)

    Vapor_Enthalpy = st.number_input("Vapor_Enthalpy [400.0 - 486.45 kJ/kg]")
    st.write(Vapor_Enthalpy)

    Enthalpy_of_Vaporization = st.number_input(
        "Enthalpy_of_Vaporization [83.0 - 173.0 kJ/kg]"
    )
    st.write(Enthalpy_of_Vaporization)

    Liquid_Specific_heat_capacity = st.number_input(
        "Liquid_Specific_heat_capacity [1010.00 - 19600.0 J/kgK]"
    )
    st.write(Liquid_Specific_heat_capacity)

    Vapor_Specific_heat_capacity = st.number_input(
        "Vapor_Specific_heat_capacity [105.00 - 2400.0 J/kgK]"
    )
    st.write(Vapor_Specific_heat_capacity)

    Critical_Temperature = st.number_input("Critical_Temperature [100.0 - 170.0 oC]")
    st.write(Critical_Temperature)

    Critical_Pressure = st.number_input("Critical_Pressure [3600.00 - 3700.0 kPa]")
    st.write(Critical_Pressure)

    Tube_Length = st.number_input("Tube_Length [125.0 - 3000.0 mm]")
    st.write(Tube_Length)

    HTC = st.button("Predict Heat Transfer Coefficient")
    if HTC:
        X_new = np.array(
            [
                [
                    Working_Fluid,
                    Tube_Geometry,
                    Internal_Diameter,
                    Saturation_Temperature,
                    Saturation_Pressure,
                    Mass_Velocity,
                    Heat_Flux,
                    Vapor_Quality,
                    Reduced_Temperature,
                    Reduced_Pressure,
                    Liquid_Density,
                    Vapor_Density,
                    Liquid_Viscosity,
                    Vapor_Viscosity,
                    Liquid_Thermal_Conductivity,
                    Vapor_Thermal_Conductivity,
                    Surface_Tension,
                    Liquid_Enthalpy,
                    Vapor_Enthalpy,
                    Enthalpy_of_Vaporization,
                    Liquid_Specific_heat_capacity,
                    Vapor_Specific_heat_capacity,
                    Critical_Temperature,
                    Critical_Pressure,
                    Tube_Length,
                ]
            ]
        )

        try:
            X_new[:, 0] = le_working_fluid.transform(X_new[:, 0])
        except KeyError as e:
            print(f"Error: {e}. This working fluid was not seen in the training data.")

        try:
            X_new[:, 1] = le_tube_geometry.transform(X_new[:, 1])
        except KeyError as e:
            print(f"Error: {e}. This tube geometry was not seen in the training data.")

        X_new = X_new.astype(float)

        X_new_scaled = scaler.transform(X_new)

        HTC = RF_model.predict(X_new_scaled)
        st.subheader(f"The estimated heat transfer coefficient is: {HTC[0]:.2f} W/m2K")
