const Koa = require('koa'),
      route = require('koa-route'),
      websockify = require('koa-websocket'),
      http = require('http'),
      app = websockify(new Koa());

app.ws.use(route.all('/', ctx => {
    // websocket作为“ctx.websocket”添加到上下文中。
    ctx.websocket.on('message', message => {
        startRequest(message, ctx);
    });
}));

function startRequest(message, ctx) {
    const net = require('net');
    const port = 8001;
    const hostname = '127.0.0.1';
    const sock = new net.Socket();
    sock.setEncoding = 'UTF-8'
    sock.connect(port, hostname, function(){
        sock.write(message)
    });
	sock.on('data', function(res){
        ctx.websocket.send(res.toString());
    });
}

// 监听端口、启动程序
app.listen(3000, err => {
    if (err) throw err;
})
