from gtts import gTTS

text = """
Hybrid vehicles are designed to combine the benefits of an internal combustion engine and an electric motor. 
The battery in a hybrid car stores energy that is used to power the electric motor during driving. 
This energy is recharged through regenerative braking, a process that converts kinetic energy into electrical energy.

There are two main types of hybrid systems: series and parallel. 
In a series hybrid, the internal combustion engine is used to generate electricity, which powers the electric motor. 
In a parallel hybrid, both the internal combustion engine and the electric motor can drive the wheels directly.

Hybrid vehicles are more efficient than traditional cars, reducing fuel consumption and emissions. 
They are especially popular in urban areas where stop-and-go traffic allows regenerative braking to maximize efficiency.
"""

audio = gTTS(text, lang='en')
audio.save("Hybrid_Vehicles_Lesson.mp3")
