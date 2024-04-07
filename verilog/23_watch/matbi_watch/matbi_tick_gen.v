`timescale 1ns / 1ps
module matbi_tick_gen(
    clk,
    reset,
	i_run_en,
	i_tick,
    o_tick_gen,
    o_cnt_val
    );
parameter P_DELAY_OUT = 0; // (default) 60
parameter P_COUNT_BIT = 6; // (default) 6b = 2^6 
parameter P_INPUT_CNT = 60; // (default) 60

input 								clk;
input 								reset;
input 								i_run_en;
input 								i_tick;
output reg							o_tick_gen;
output [P_COUNT_BIT-1:0]			o_cnt_val;

reg [P_COUNT_BIT-1:0]		r_cnt_val;

	always @(posedge clk) begin
	    if(reset) begin
			r_cnt_val <= {P_COUNT_BIT{1'b0}};
			o_tick_gen <= 1'b0;
	    end else if (i_run_en & i_tick) begin
			if(r_cnt_val == P_INPUT_CNT -1) begin
				r_cnt_val <= 0;
				o_tick_gen <= 1'b1;
			end else begin
				r_cnt_val <= r_cnt_val + 1'b1;
			end
	    end else begin
			o_tick_gen <= 1'b0;
		end
	end

// General cycle delay logic
genvar gi;
generate
	if(P_DELAY_OUT == 0) begin // bypass
		assign o_cnt_val = r_cnt_val;
	end else if (P_DELAY_OUT == 1) begin // 1 cycle delay
		reg [P_COUNT_BIT-1:0]		r_cnt_val_d;
		always @(posedge clk) begin
			r_cnt_val_d <= r_cnt_val;
		end
		assign o_cnt_val = r_cnt_val_d;
	end else begin  // cycle > 1
		reg [P_COUNT_BIT-1:0]		r_cnt_val_d [P_DELAY_OUT-1:0];
		always @(posedge clk) begin
			r_cnt_val_d[0] <= r_cnt_val;
		end
		for(gi = 1; gi < P_DELAY_OUT; gi = gi + 1) begin : gen_delay
			always @(posedge clk) begin
				r_cnt_val_d[gi] <= r_cnt_val_d[gi-1];
			end
		end
		assign o_cnt_val = r_cnt_val_d[P_DELAY_OUT-1];
	end
endgenerate

endmodule
