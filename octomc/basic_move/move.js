bot.setControlState('forward', true); // 开始向前移动
setTimeout(() => {
  bot.setControlState('forward', false); // 停止向前移动
}, 2500);