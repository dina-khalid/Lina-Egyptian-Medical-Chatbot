from main import chat
#
# page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))
#
#
# if page == "Predict":
#     showHomePage()
# # else:
# #     showExplorePage()
import discord
import os
from neuralintents import GenericAssistant


client = discord.Client()

TOKEN="DISCORD TOKEN"

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content == "ping":
        await message.channel.send('pong')



    if message.content.startswith("يا لينا"):
        print(message.content)
        response = chat(message.content[8:])
        await message.channel.send(response)


client.run(TOKEN)
