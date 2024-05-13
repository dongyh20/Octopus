
async function perceive(block_name) {
    const blockByName = mcData.blocksByName[block_name];
    const block = bot.findBlock({ // 有可能找到该方块
        matching: (block) => block.name === blockByName.name,
        maxDistance: 32
    });
    if (block) {
            if (bot.canSeeBlock(block)) {
              bot.chat(`I can see ${block.name}`);
            } 
            else {
              bot.chat(`I can't see ${block.name}`);
            }
        }}

await perceive("oak_log")