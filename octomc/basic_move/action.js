const [rotation, distance] =await find("sand") //For GPT pipeline to get the gt rot and dis
await look_around()
await teleport(rotation,distance)
await delay(5000)
await look_around()
