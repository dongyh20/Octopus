const angles = [0, 90, 180, 270];
const PI = Math.PI

function get_see_SurroundingBlocks(bot, x_distance, y_distance, z_distance,index) {
  const surroundingBlocks = new Set();
  if (index==0){  // -z
      // bot.chat(`CASE1`)
      for (let x = -x_distance; x <= x_distance; x++) {
          for (let y = -y_distance; y <= y_distance; y++) {
              for (let z = -z_distance; z <= 0; z++) { 
                  const block = bot.blockAt(bot.entity.position.offset(x, y, z));
                  if (block && block.type !== 0) {
                      if (bot.canSeeBlock(block)){ //if the bot can see the block
                          // bot.chat(`${}`)
                          surroundingBlocks.add([block.name,cal_distance_by_block(bot,block)]);
                      }
                      
                  }
              }
          }
      }
  }
  if (index==1){  //-x 
      // bot.chat(`CASE2`)
      for (let x = -x_distance; x <= 0; x++) {
          for (let y = -y_distance; y <= y_distance; y++) {
              for (let z =-z_distance; z <= z_distance; z++) { 
                  const block = bot.blockAt(bot.entity.position.offset(x, y, z));
                  if (block && block.type !== 0) {
                      if (bot.canSeeBlock(block)){ //if the bot can see the block
                          surroundingBlocks.add([block.name,cal_distance_by_block(bot,block)]);
                      }
                      
                  }
              }
          }
      }
  }

  if (index==2){  // +z
      // bot.chat(`CASE3`)
      for (let x = -x_distance; x <= x_distance; x++) {
          for (let y = -y_distance; y <= y_distance; y++) {
              for (let z =0; z <= z_distance; z++) { 
                  const block = bot.blockAt(bot.entity.position.offset(x, y, z));
                  if (block && block.type !== 0) {
                      if (bot.canSeeBlock(block)){ //if the bot can see the block
                          surroundingBlocks.add([block.name,cal_distance_by_block(bot,block)]);
                      }
                      
                  }
              }
          }
      }
  }

  if (index==3){  //+x 
      // bot.chat(`CASE4`)
      for (let x = 0; x <= x_distance; x++) {
          for (let y = -y_distance; y <= y_distance; y++) {
              for (let z =-z_distance; z <= z_distance; z++) { 
                  const block = bot.blockAt(bot.entity.position.offset(x, y, z));
                  if (block && block.type !== 0) {
                      if (bot.canSeeBlock(block)){ //if the bot can see the block
                          surroundingBlocks.add([block.name,cal_distance_by_block(bot,block)]);
                      }
                      
                  }
              }
          }
      }
  }
  // console.log(surroundingBlocks);
  return surroundingBlocks;
}
// function get_see_SurroundingBlocks(bot, x_distance, y_distance, z_distance) {
//   const surroundingBlocks = new Set();
//   if ((Math.sin(bot.entity.yaw))>=0 && (Math.cos(bot.entity.yaw))>=0){  //-x -z
//       // bot.chat(`CASE1`)
//       for (let x = -x_distance; x <= 0; x++) {
//           for (let y = -y_distance; y <= y_distance; y++) {
//               for (let z = -z_distance; z <= 0; z++) { 
//                   const block = bot.blockAt(bot.entity.position.offset(x, y, z));
//                   if (block && block.type !== 0) {
//                       if (bot.canSeeBlock(block)){ //if the bot can see the block
//                           // bot.chat(`${}`)
//                           surroundingBlocks.add([block.name,cal_distance_by_block(bot,block)]);
//                       }
                      
//                   }
//               }
//           }
//       }
//   }
//   if ((Math.sin(bot.entity.yaw))>=0 && (Math.cos(bot.entity.yaw))<=0){  //-x +z
//       // bot.chat(`CASE2`)
//       for (let x = -x_distance; x <= 0; x++) {
//           for (let y = -y_distance; y <= y_distance; y++) {
//               for (let z =0; z <= z_distance; z++) { 
//                   const block = bot.blockAt(bot.entity.position.offset(x, y, z));
//                   if (block && block.type !== 0) {
//                       if (bot.canSeeBlock(block)){ //if the bot can see the block
//                           surroundingBlocks.add([block.name,cal_distance_by_block(bot,block)]);
//                       }
                      
//                   }
//               }
//           }
//       }
//   }

//   if ((Math.sin(bot.entity.yaw))<=0 && (Math.cos(bot.entity.yaw))<=0){  //+x +z
//       // bot.chat(`CASE3`)
//       for (let x = 0; x <= x_distance; x++) {
//           for (let y = -y_distance; y <= y_distance; y++) {
//               for (let z =0; z <= z_distance; z++) { 
//                   const block = bot.blockAt(bot.entity.position.offset(x, y, z));
//                   if (block && block.type !== 0) {
//                       if (bot.canSeeBlock(block)){ //if the bot can see the block
//                           surroundingBlocks.add([block.name,cal_distance_by_block(bot,block)]);
//                       }
                      
//                   }
//               }
//           }
//       }
//   }

//   if ((Math.sin(bot.entity.yaw))<=0 && (Math.cos(bot.entity.yaw))>=0){  //+x -z
//       // bot.chat(`CASE4`)
//       for (let x = 0; x <= x_distance; x++) {
//           for (let y = -y_distance; y <= y_distance; y++) {
//               for (let z =-z_distance; z <= 0; z++) { 
//                   const block = bot.blockAt(bot.entity.position.offset(x, y, z));
//                   if (block && block.type !== 0) {
//                       if (bot.canSeeBlock(block)){ //if the bot can see the block
//                           surroundingBlocks.add([block.name,cal_distance_by_block(bot,block)]);
//                       }
                      
//                   }
//               }
//           }
//       }
//   }
//   // console.log(surroundingBlocks);
//   return surroundingBlocks;
// }

// Utility Functions:
function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
async function screenshot(){
    const robot = require('robotjs');
    robot.keyTap('f2');
  }
async function disabled_falldamage(){
  robot.chat("/gamerule fallDamage false")
}
function cal_distance(currentSpotX,currentSpotY,currentSpotZ,targetX,targetY,targetZ){
  const deltaX = targetX - currentSpotX;
  const deltaY = targetY - currentSpotY;
  const deltaZ = targetZ - currentSpotZ;
  
  const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
  return distance
}
function cal_distance_by_block(bot,block){
  const current_spot_x=bot.entity.position.x
  const current_spot_y=bot.entity.position.y 
  const current_spot_z=bot.entity.position.z 
  const x = block.position.x;
  const y = block.position.y;
  const z = block.position.z;
  return cal_distance(current_spot_x,current_spot_y,current_spot_z,x,y,z).toFixed(2)
}

function findwhichimage(rotation){
}

async function find(block_name){
  const current_spot_x=bot.entity.position.x
  const current_spot_y=bot.entity.position.y 
  const current_spot_z=bot.entity.position.z 
  let can_see=false
  await bot.chat(`Current spot at X: ${current_spot_x}, Y: ${current_spot_y}, Z: ${current_spot_z}`);
  const blockByName = mcData.blocksByName[block_name];
try{
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
  await bot.chat(`the ${block.name} at ${x} ${y} ${z}`)
  await bot.lookAt(block.position)
  delay(2000)
  const rotation=bot.entity.yaw
  const distance=cal_distance(current_spot_x,current_spot_y,current_spot_z,x,y,z)
  bot.chat(`the distance between the bot and the block is ${distance}`)
  bot.chat(`${rotation}`)
  // degree=rad_to_degree(rotation)
  // bot.chat(`the angle towards ${block_name} is ${degree}`)
  
  return [block,rotation,distance]
}}
catch (error){
  bot.chat(`${error}`)
}
}

// Action List Functions:
function teleport(rot, dis) {
  return new Promise(async (resolve, reject) => {
    try {
      //when the rotation is PI, the minecraft shows 0
      await bot.look(rot, 0);
      delay(2000);
      bot.chat(`now look at ${rot}`);
      x = bot.entity.position.x;
      y = bot.entity.position.y;
      z = bot.entity.position.z;
      const dz = -dis * Math.cos(rot);
      const dx = -dis * Math.sin(rot);
      x = x + dx;
      z = z + dz;
      bot.chat(`/tp ${x} ${y + 3} ${z}`);
      delay(2000);
      resolve(); 
    } catch (error) {
      reject(error); 
    }
  });
}


async function look_around() {
  // yaw = bot.entity.yaw
  yaw = 0
  for (let index = 0; index < 4; index++) {
    // bot.chat(`${bot.entity.yaw}`)
    // blockinfo=get_see_SurroundingBlocks(bot, Math.abs(Math.floor(100*Math.sin(bot.entity.yaw))), 2, Math.abs(Math.floor(100*Math.cos(bot.entity.yaw))))
    blockinfo=get_see_SurroundingBlocks(bot, 20,3,20,index)
    bot.chat(`pic${index+1}:{${Array.from(blockinfo)}},yaw=${yaw.toFixed(2)}`)
    await bot.look(yaw, 0);
    await delay(1000);
    await screenshot();
    // 停留一秒钟
    yaw=yaw+Math.PI/2;
  }
}

async function demo_pipeline(block_name) {
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
  await bot.chat(`the ${block.name} at ${x} ${y} ${z}`)
  await bot.lookAt(block.position)
  await bot.chat('look success')
  await new_look_around()
  // await bot.chat(`/tp ${x} ${y+10} ${z}`) teleport version
  await bot.lookAt(block.position) //look at the target
  // await super_explore(bot,block.position)
  const woodLogBlock=await super_explore(bot, block.position, 60, () => {
    const foundLog = bot.findBlock({
      matching: tar_block => tar_block.name==block.name,
      maxDistance: 2
    });
    return foundLog;
  });
  // let goal = new GoalNear(x, y, z);
  // await bot.pathfinder.setGoal(goal)
  await mineBlock(bot, woodLogBlock.name, 1);
  await new_look_around()
  bot.chat("mission success!!!")
  return 
  // await bot.chat(`start teleport`);
  // await bot.chat(`Found ${block_name} at X: ${x}, Y: ${y}, Z: ${z}`);
  // await bot.chat(`/tp ${x} ${y} ${z}`)
  // await bot.chat(`finish!!!!`)
  // return can_see,block}
}}
catch (error){
  bot.chat(`${error}`)
}
}

async function super_explore(
  bot,
  direction,
  maxTime = 60,
  callback = () => {
      return false;
  }
) {
  if (typeof maxTime !== "number") {
      throw new Error("maxTime must be a number");
  }
  if (typeof callback !== "function") {
      throw new Error("callback must be a function");
  }
  const test = callback();
  if (test) {
      bot.chat("Explore success.");
      return Promise.resolve(test);
  }
  if (direction.x === 0 && direction.y === 0 && direction.z === 0) {
      throw new Error("direction cannot be 0, 0, 0");
  }
  maxTime = Math.min(maxTime, 1200);
  return new Promise((resolve, reject) => {
      const dx = direction.x;
      const dy = direction.y;
      const dz = direction.z;

      let explorationInterval;
      let maxTimeTimeout;

      const cleanUp = () => {
          clearInterval(explorationInterval);
          clearTimeout(maxTimeTimeout);
          bot.pathfinder.setGoal(null);
      };

      const explore = () => {
          const x =dx
          const y =dy
          const z =dz
          let goal = new GoalNear(x, y, z);
          if (dy === 0) {
              goal = new GoalNearXZ(x, z);
          }
          bot.pathfinder.setGoal(goal);

          try {
              const result = callback();
              if (result) {
                  cleanUp();
                  bot.chat("Explore success.");
                  resolve(result);
              }
          } catch (err) {
              cleanUp();
              reject(err);
          }
      };

      explorationInterval = setInterval(explore, 2000);

      maxTimeTimeout = setTimeout(() => {
          cleanUp();
          bot.chat("Max exploration time reached");
          resolve(null);
      }, maxTime * 1000);
  });
}

async function verify(block) {
  const bot_x=bot.entity.position.x
  const bot_y=bot.entity.position.y 
  const bot_z=bot.entity.position.z 
  const x = block.position.x;
  const y = block.position.y;
  const z = block.position.z;
  bot.chat(`${bot_x},${bot_y},${bot_z}`)
try{
  dis=cal_distance(bot_x,bot_y,bot_z,x,y,z)
  if (dis<50){
    return true
  }
  else {
    return false
  }
}
catch (error){
  bot.chat(`${error}`)
}
}


