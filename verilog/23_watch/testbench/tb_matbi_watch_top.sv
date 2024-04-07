`timescale 1ns / 1ps
module tb_matbi_watch_top;
localparam P_COUNT_BIT = 30; // (default) 30b, under 1GHz. 2^30 = 1073741824
localparam P_SEC_BIT	 = 6; // 2^6 = 64
localparam P_MIN_BIT	 = 6; // 2^6 = 64 
localparam P_HOUR_BIT = 5; // 2^5 = 32 
//DUT Port List
reg clk, reset;
reg i_run_en;
reg [P_COUNT_BIT-1:0]	i_freq;
wire [P_SEC_BIT-1:0]	o_sec;
wire [P_MIN_BIT-1:0]	o_min;
wire [P_HOUR_BIT-1:0]	o_hour;

// clk gen
always
    #5 clk = ~clk;

/////// Main ////////
initial begin
//initialize value
	i_freq		<= 10;
	assert(i_freq <= 1000*1000*1000); // under 1GHz
$display("initialize value [%d]", $time);
    reset 		<= 0;
    clk     	<= 0;
	i_run_en 	<= 0;
// reset gen
$display("Reset! [%d]", $time);
# 100
    reset 		<= 1;
# 10
    reset 		<= 0;
    i_run_en 	<= 1;
# 10
@(posedge clk);
$display("Start! [%d]", $time);
# 10000000
$display("Finish! [%d]", $time);
	i_run_en	<= 0;
$finish;
end

// Call DUT
matbi_watch_top 
# (
	.P_COUNT_BIT	(P_COUNT_BIT),
	.P_SEC_BIT	 	(P_SEC_BIT	),
	.P_MIN_BIT	 	(P_MIN_BIT	),
	.P_HOUR_BIT		(P_HOUR_BIT	)
) u_matbi_watch_top(
	.clk 				(clk),
	.reset 				(reset),
	.i_run_en			(i_run_en),
	.i_freq				(i_freq),
    .o_sec				(o_sec ),
    .o_min				(o_min ),
    .o_hour				(o_hour)
);

endmodule
