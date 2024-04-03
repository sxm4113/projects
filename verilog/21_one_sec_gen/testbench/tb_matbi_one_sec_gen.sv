`timescale 1ns/1ps
module tb_matbi_one_sec_gen;
localparam COUNT_BIT = 30;
reg clk, reset;
reg i_run_en;
reg [COUNT_BIT-1:0] i_freq;
wire o_one_Sec_tick;

always #5 clk = ~clk;

initial begin
    reset <=0;
    clk <=0;
    i_run_en <=0;
    #100
        reset <= 1;
    #10 
        reset <= 0;
        i_run_en <= 1;
        i_freq <= 100;
    #10
        @(posedge clk)
        $display ("start! [%d]", $time);
    #100000
        $display ("finish! [%d]", $time);
        i_run_en <=0;
        $finish;
end

matbi_one_sec_gen
# (
    .COUNT_BIT (COUNT_BIT)
) u_matbi_one_sec_gen(
    	.clk 				(clk),
	.reset 				(reset),
	.i_run_en			(i_run_en),
	.i_freq				(i_freq),
	.o_one_sec_tick 	(o_one_sec_tick)
); 

reg [5:0] r_sec;
parameter ONE_MIN_IN_SEC = 60;
always @(posedge clk) begin
    if (reset) begin
        r_sec <= 6'b0;
    end else if (o_one_sec_tick) begin
        if (r_sec == ONE_MIN_IN_SEC-1) begin
            r_sec <=0;
        end else begin
            r_sec <= r_sec + 1'b1;
        end
    end
end
 
endmodule