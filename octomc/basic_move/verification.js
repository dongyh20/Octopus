async function cal_distance(currentSpotX,currentSpotY,currentSpotZ,targetX,targetY,targetZ){
    const deltaX = targetX - currentSpotX;
    const deltaY = targetY - currentSpotY;
    const deltaZ = targetZ - currentSpotZ;
    
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
    return distance
  }

async function verify(block_name) {
    const current_spot_x=bot.entity.position.x
    const current_spot_y=bot.entity.position.y 
    const current_spot_z=bot.entity.position.z 
    let can_see=false
    await bot.chat(`Current spot at X: ${current_spot_x}, Y: ${current_spot_y}, Z: ${current_spot_z}`);
    const blockByName = mcData.blocksByName[block_name];
  try{
    // const block = bot.nearestEntity((entity) => entity.name === 'Chicken' && bot.entity.position.distanceTo(entity.position) < 54);
    const block = bot.findBlock({ 
      matching: (block) => block.name === blockByName.name,
      maxDistance: 64
    });
  
    
    if (block){
        if (bot.canSeeBlock(block)) {
          await bot.chat(`I can see ${block.name}`);
          can_see=true
        } 
        else {
          await bot.chat(`I can't see ${block.name}`);
        }
  
        // 停留一秒钟
        await delay(500); 
      }
    else {
      await bot.chat(`Did not find ${block_name}.`);
      return
    }
    if (can_see==true)
  {  const x = block.position.x;
    const y = block.position.y;
    const z = block.position.z;
    await bot.look(0,0)
    await bot.chat(`the ${block.name} at ${x} ${y} ${z}`)
    await bot.lookAt(block.position)
    await bot.chat('look success')
    // await new_look_around()
    // await bot.chat(`/tp ${x} ${y+10} ${z}`) teleport version
    delay(5000)
    await bot.lookAt(block.position) //look at the target
    delay(5000)
    const yaw=bot.entity.yaw
    bot.chat(`the yaw is ${yaw}`)
    delay(500)
    await bot.look(yaw+PI,0)
    delay(5000)
    await bot.look(yaw,0)
  }}
  catch (error){
    bot.chat(`${error}`)
  }
  }