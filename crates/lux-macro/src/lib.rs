use quote::quote;
use syn::{ItemStruct, parse_macro_input};

#[proc_macro_derive(Component)]
pub fn derive_component(_item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(_item as ItemStruct);
    let name = input.ident;
    (quote! {
        impl Component for #name {}
    }).into()
}